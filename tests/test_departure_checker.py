import unittest
from copy import deepcopy
import importlib

import cv2
import numpy as np

from dash_referee import DashConfig, DashReferee, Motion, paint_geometry
from departure_checker import DepartureChecker


SHAPE = (180, 260)
IDENTITY = Motion(np.eye(3), True, 0, "test")


def lanes(intervals=((50, 150),), lane_id=1, matrix=None, reverse=False, y_range=(90, 101)):
    mask = np.zeros(SHAPE, np.uint8)
    for start, end in intervals:
        mask[y_range[0]:y_range[1], start:end + 1] = 255
    if matrix is not None:
        mask = cv2.warpPerspective(mask, matrix, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_NEAREST)
    result = paint_geometry(mask, np.int32(mask > 0) * lane_id, [lane_id], 50)
    if reverse:
        for lane in result:
            lane["centerline"] = tuple(reversed(lane["centerline"]))
    return result


def wheel(x=100, y=80, identity=1):
    return dict(track_id=identity, bbox=(x - 15, y - 25, x + 15, y + 2),
                keypoint=(x, y), kp_valid=True, conf=0.95)


class DepartureTests(unittest.TestCase):
    def setUp(self):
        self.checker = DepartureChecker()

    def step(self, point, timestamp, paint=None, motion=IDENTITY, identity=1):
        return self.checker.update(lanes() if paint is None else paint,
                                   [wheel(*point, identity)], motion, timestamp)

    def test_downward_crossing_is_out(self):
        self.step((100, 80), 0)
        self.assertFalse(self.step((100, 95), 0.05))
        event = self.step((100, 110), 0.1)[0]
        self.assertGreater(event["keypoint_y"], event["paint_bottom_y"] + event["required_margin_px"])
        self.assertIn(1, self.checker.active)

    def test_return_to_upper_normal_side_is_not_out(self):
        self.step((100, 80), 0)
        self.step((100, 95), 0.05)
        self.assertFalse(self.step((100, 80), 0.1))
        self.assertNotIn(1, self.checker.active)

    def test_below_to_above_is_not_a_new_out_event(self):
        self.step((100, 110), 0)
        self.step((100, 95), 0.05)
        self.assertFalse(self.step((100, 80), 0.1))
        self.assertFalse(self.checker.active)

    def test_end_entry_then_below_is_out_without_opposite_side_history(self):
        # test_1 front wheel: first seen on the unpainted extension, then on paint, then below.
        self.step((170, 95), 0)
        self.step((140, 95), 0.05)
        self.assertEqual(len(self.step((100, 110), 0.1)), 1)

    def test_paint_contact_then_gap_is_normal_not_a_delayed_crossing(self):
        # test_1 rear wheel: do not turn a later normal-side sign change in a gap into out.
        self.step((140, 80), 0)
        self.step((145, 95), 0.05)
        self.assertFalse(self.step((200, 110), 0.1))
        self.assertFalse(self.checker.active)

    def test_past_endpoint_without_contact_is_normal(self):
        self.step((170, 80), 0)
        self.assertFalse(self.step((170, 110), 1 / 15))

    def test_longitudinal_end_exit_is_normal(self):
        self.step((100, 80), 0)
        self.step((140, 95), 0.05)
        self.assertFalse(self.step((170, 95), 0.1))

    def test_bbox_overlapping_paint_does_not_replace_contact_x(self):
        self.assertFalse(self.step((160, 110), 0))

    def test_between_two_actual_dashes_is_normal(self):
        paint = lanes(((20, 70), (130, 180)))
        self.step((100, 80), 0, paint)
        self.assertFalse(self.step((100, 110), 1 / 15, paint))

    def test_gap_between_vertical_dashes_is_not_below_paint_departure(self):
        mask = np.zeros(SHAPE, np.uint8)
        mask[20:71, 90:101] = 255
        mask[130:171, 90:101] = 255
        paint = paint_geometry(mask, np.int32(mask > 0), [1], 50)
        self.assertFalse(self.step((95, 100), 0, paint))

    def test_diagonal_dash_endpoint_is_not_extended_downward(self):
        mask = np.zeros(SHAPE, np.uint8)
        cv2.fillPoly(mask, [np.int32([[50, 70], [150, 120], [145, 130], [45, 80]])], 255)
        paint = paint_geometry(mask, np.int32(mask > 0), [1], 50)
        self.assertFalse(self.step((150, 140), 0, paint))

    def test_small_edge_jitter_stays_within_margin(self):
        for index, y in enumerate([95, 100, 101, 99, 101]):
            self.assertFalse(self.step((100, y), index / 30))

    def test_out_does_not_require_200ms_contact(self):
        self.step((100, 80), 0)
        self.assertEqual(len(self.step((100, 110), 1 / 15)), 1)

    def test_already_below_known_paint_can_be_judged_from_current_geometry(self):
        self.assertEqual(len(self.step((100, 110), 0)), 1)

    def test_starting_on_paint_then_below_can_be_judged(self):
        self.step((100, 95), 0)
        self.assertEqual(len(self.step((100, 110), 0.1)), 1)

    def test_previously_observed_paint_recovers_occlusion(self):
        self.step((100, 80), 0)
        self.assertEqual(len(self.step((100, 110), 0.1, [])), 1)

    def test_never_observed_paint_cannot_create_out(self):
        self.step((100, 80), 0, [])
        self.assertFalse(self.step((100, 110), 0.1, []))

    def test_unrelated_paint_far_above_wheel_is_not_its_boundary(self):
        self.assertFalse(self.step((100, 155), 0))

    def test_closest_paint_is_chosen_before_out_classification(self):
        paint = lanes(lane_id=1) + lanes(lane_id=2, y_range=(105, 116))
        self.assertFalse(self.step((100, 110), 0, paint))

    def test_one_event_while_currently_out(self):
        self.assertEqual(len(self.step((100, 110), 0)), 1)
        self.assertFalse(self.step((100, 115), 0.05))
        self.assertIn(1, self.checker.active)

    def test_history_does_not_keep_gap_red(self):
        self.step((100, 110), 0)
        self.step((175, 110), 0.05)
        self.assertFalse(self.checker.active)
        self.assertIn(1, self.checker.departed)

    def test_overlapping_memories_emit_one_wheel_event(self):
        self.step((100, 95), 0)
        duplicate = deepcopy(next(iter(self.checker.anchors.values())))
        duplicate.identity = 99
        self.checker.anchors[99] = duplicate
        self.assertEqual(len(self.step((100, 110), 0.1, [])), 1)
        self.assertFalse(self.step((100, 115), 0.2, []))

    def test_registration_failure_cannot_use_old_paint(self):
        self.step((100, 80), 0)
        self.assertFalse(self.step((100, 110), 0.1, [], Motion(reason="failed")))

    def test_current_observed_paint_does_not_require_registration(self):
        self.assertEqual(len(self.step((100, 110), 0, motion=Motion(reason="failed"))), 1)

    def test_initial_paint_is_saved_before_next_frame_occlusion(self):
        self.step((100, 80), 0, motion=Motion(reason="initial"))
        self.assertEqual(len(self.step((100, 110), 0.1, [])), 1)

    def test_camera_translation_alone_is_not_out(self):
        self.step((100, 80), 0)
        matrix = np.array([[1, 0, 10], [0, 1, 25], [0, 0, 1]], float)
        self.assertFalse(self.step((110, 105), 0.1, lanes(matrix=matrix),
                                   Motion(matrix, True, 0, "translation")))

    def test_downward_crossing_with_camera_translation_is_out(self):
        self.step((100, 80), 0)
        matrix = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], float)
        self.assertEqual(len(self.step((110, 115), 0.1, lanes(matrix=matrix),
                                       Motion(matrix, True, 0, "translation"))), 1)

    def test_camera_rotation_alone_is_not_out(self):
        self.step((100, 80), 0)
        matrix = np.vstack([cv2.getRotationMatrix2D((130, 90), 20, 1), [0, 0, 1]])
        point = cv2.perspectiveTransform(np.float32([[[100, 80]]]), matrix)[0, 0]
        self.assertFalse(self.step(tuple(point), 0.1, lanes(matrix=matrix),
                                   Motion(matrix, True, 0, "rotation")))

    def test_pca_sign_does_not_change_screen_down_rule(self):
        self.step((100, 95), 0)
        self.assertEqual(len(self.step((100, 110), 0.05, lanes(reverse=True))), 1)

    def test_changed_segmentation_id_does_not_repeat_same_out(self):
        self.step((100, 110), 0, lanes(lane_id=1))
        self.assertFalse(self.step((100, 110), 0.1, lanes(lane_id=99)))
        self.assertEqual(len(self.checker.anchors), 1)

    def test_new_wheel_is_judged_independently_with_known_down_side(self):
        self.step((100, 80), 0, identity=1)
        event = self.step((100, 110), 0.1, identity=2)[0]
        self.assertEqual(event["wheel_id"], 2)

    def test_reset_clears_history_and_allows_current_out_to_be_judged_again(self):
        self.step((100, 110), 0)
        self.checker.reset_wheels()
        self.assertFalse(self.checker.departed)
        self.assertEqual(len(self.step((100, 110), 0.1)), 1)

    def test_memory_without_wheel_proximity_expires(self):
        self.step((100, 80), 0)
        self.assertFalse(self.step((100, 110), 1.3, []))

    def test_recent_near_wheel_observation_has_bounded_grace(self):
        self.step((100, 95), 0)
        self.assertEqual(len(self.step((100, 110), 1.3, [])), 1)
        self.assertFalse(self.step((100, 110), 2.01, []))
        self.assertFalse(self.checker.active)

    def test_repeated_predictions_do_not_refresh_model_lifetime(self):
        self.step((100, 95), 0)
        for timestamp in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8):
            self.step((100, 95), timestamp, [])
        self.assertFalse(self.step((100, 110), 2.01, []))

    def test_large_registration_uncertainty_discards_memory(self):
        self.step((100, 95), 0)
        self.assertFalse(self.step((100, 110), 0.1, [],
                                   Motion(np.eye(3), True, 5, "uncertain")))


class LocalPaintValidationTests(unittest.TestCase):
    def setUp(self):
        self.checker = DepartureChecker()
        self.paint = lanes()
        self.checker.update(self.paint, [wheel(y=95)], IDENTITY, 0)

    def labels(self, start=88, stop=99, identity=1, x_range=(50, 151)):
        result = np.zeros(SHAPE, np.int32)
        result[start:stop, x_range[0]:x_range[1]] = identity
        return result

    def evaluate(self, labels, point=(100, 104), error=3, now=0.05):
        return self.checker.update([], [wheel(*point)], Motion(np.eye(3), True, error, "test"),
                                   now, observed_labels=labels)

    def test_fresh_visible_paint_revalidates_vertical_edge(self):
        events = self.evaluate(self.labels())
        self.assertEqual(len(events), 1)
        anchor = next(iter(self.checker.anchors.values()))
        self.assertEqual(anchor.full_time, 0)
        self.assertEqual(anchor.variance, 9)
        self.assertEqual(anchor.matrix[1, 2], -2)
        self.assertLess(events[0]["edge_error_px"], events[0]["model_error_px"])
        self.assertGreater(events[0]["verified_columns"], 20)

    def test_no_observation_does_not_lower_uncertainty(self):
        self.assertFalse(self.evaluate(None))

    def test_prediction_pixels_cannot_validate_themselves(self):
        self.assertFalse(self.evaluate(np.zeros(SHAPE, np.int32)))

    def test_observation_under_wheel_is_excluded(self):
        self.assertFalse(self.evaluate(self.labels(x_range=(85, 116))))

    def test_other_paint_identity_is_not_used_for_validation(self):
        self.assertFalse(self.evaluate(self.labels(identity=99)))

    def test_narrow_occlusion_fragment_is_not_a_full_paint_width(self):
        self.assertFalse(self.evaluate(self.labels(start=90, stop=94)))

    def test_short_fragment_does_not_prove_alignment(self):
        self.assertFalse(self.evaluate(self.labels(x_range=(50, 62))))

    def test_large_correction_is_rejected(self):
        self.assertFalse(self.evaluate(self.labels(start=81, stop=92)))

    def test_held_out_columns_must_agree_with_estimated_correction(self):
        labels = self.labels(start=90, stop=101)
        labels[:, 51:151:2] = 0
        labels[93:104, 51:151:2] = 1
        self.assertFalse(self.evaluate(labels))

    def test_same_paint_can_continue_out_while_still_below_validated_edge(self):
        self.checker.update(self.paint, [wheel(y=110)], IDENTITY, 0.01)
        self.evaluate(self.labels(start=90, stop=101), point=(100, 101), error=1, now=0.05)
        self.assertIn(1, self.checker.active)
        self.assertEqual(self.checker.active[1]["decision_rule"], "verified_continuation")
        self.evaluate(self.labels(start=90, stop=101), point=(100, 100), error=0, now=0.1)
        self.assertFalse(self.checker.active)

    def test_continuation_never_extends_paint_into_gap(self):
        self.checker.update(self.paint, [wheel(y=110)], IDENTITY, 0.01)
        self.evaluate(self.labels(start=90, stop=101), point=(175, 110), error=1, now=0.05)
        self.assertFalse(self.checker.active)
        self.evaluate(self.labels(start=90, stop=101), point=(100, 101), error=0, now=0.1)
        self.assertFalse(self.checker.active)

    def test_local_verification_does_not_renew_hidden_shape_lifetime(self):
        for now in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8):
            self.evaluate(self.labels(start=90, stop=101), point=(100, 95), error=0, now=now)
        self.assertFalse(self.evaluate(self.labels(start=90, stop=101), point=(100, 110), error=0, now=2.01))


class BinaryIntegrationTests(unittest.TestCase):
    def test_unobserved_frame_gap_does_not_count_as_continuous_intrusion(self):
        engine = DashReferee(DashConfig(entry_hold_sec=0.2))
        frame = np.zeros((*SHAPE, 3), np.uint8)
        first = engine.update(frame, lanes(), [wheel(y=95)], 0, motion_override=IDENTITY)
        later = engine.update(frame, lanes(), [wheel(y=95)], 0.3, motion_override=IDENTITY)
        self.assertEqual(first["wheels"][0]["track_id"], later["wheels"][0]["track_id"])
        self.assertTrue(later["states"][0]["violated"])
        self.assertFalse(later["states"][0]["intrusion_confirmed"])

    def test_out_is_independent_of_intrusion_hold(self):
        engine = DashReferee(DashConfig(entry_hold_sec=0.2))
        frame = np.zeros((*SHAPE, 3), np.uint8)
        engine.update(frame, lanes(), [wheel(y=80)], 0, motion_override=IDENTITY)
        engine.update(frame, lanes(), [wheel(y=95)], 1 / 30, motion_override=IDENTITY)
        result = engine.update(frame, lanes(), [wheel(y=110)], 2 / 30, motion_override=IDENTITY)
        state = result["states"][0]
        self.assertFalse(state["intrusion_confirmed"])
        self.assertTrue(state["departure_active"])
        self.assertTrue(state["confirmed"])
        self.assertNotIn("phase", state)

    def test_current_gap_is_green_even_with_recorded_out_in_both_entrypoints(self):
        engine = DashReferee(DashConfig(entry_hold_sec=0.2))
        frame = np.zeros((*SHAPE, 3), np.uint8)
        for index, point in enumerate([(100, 80), (100, 95), (100, 110), (100, 110),
                                       (120, 110), (140, 110), (160, 110)]):
            result = engine.update(frame, lanes(), [wheel(*point)], index * 0.05,
                                   motion_override=IDENTITY)
        state = result["states"][0]
        self.assertEqual(state["wheel_id"], 1)
        self.assertTrue(state["departure_recorded"])
        self.assertFalse(state["departure_active"])
        self.assertFalse(state["confirmed"])
        for name in ("main", "main_capture_dual_youtube"):
            module = importlib.import_module(name)
            rendered = module.draw_frame(frame, [], result["wheels"], result["states"], False)
            x1, y1, _, _ = result["wheels"][0]["bbox"]
            self.assertEqual(tuple(rendered[y1, x1]), (0, 255, 0), name)


if __name__ == "__main__":
    unittest.main()
