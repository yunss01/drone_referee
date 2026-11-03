import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from dash_referee import (ContactEpisode, DashConfig, DashReferee, Motion, RoadMotion,
                          PaintMemory, WheelAssociation, supports_point)


SHAPE = (180, 260)
FRAME = np.zeros((*SHAPE, 3), np.uint8)
IDENTITY = Motion(np.eye(3), True, 0.1, "test")


def paint(*intervals):
    mask = np.zeros(SHAPE, np.uint8)
    for a, b in intervals:
        mask[90:101, a:b + 1] = 255
    return mask


def lanes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [dict(mask_bin=mask, contours=contours)] if contours else []


def wheel(point=(100, 95), box=(80, 65, 125, 115)):
    return dict(bbox=box, keypoint=point, kp_valid=True, conf=0.9)


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.config = DashConfig(entry_hold_sec=0, exit_hold_sec=0.05,
                                 memory_sec=0.3, history_sec=0.8, legacy_contact_events=True)
        self.engine = DashReferee(self.config)

    def step(self, mask, wheels, t, motion=IDENTITY):
        return self.engine.update(FRAME, lanes(mask), wheels, t, motion_override=motion)

    def test_occluded_known_paint_is_recovered(self):
        full = paint((50, 105))
        self.step(full, [], 0)
        hidden = full.copy(); hidden[:, 80:126] = 0
        result = self.step(hidden, [wheel()], 0.1)
        self.assertTrue(result["states"][0]["confirmed"])
        self.assertEqual(result["states"][0]["contact_source"], "memory_assisted")
        self.assertGreater(result["recovered_pixels"], 0)

    def test_completely_hidden_dash_uses_only_prior_observation(self):
        self.step(paint((80, 110)), [], 0)
        result = self.step(paint(), [wheel()], 0.1)
        self.assertTrue(result["states"][0]["violated"])

    def test_never_seen_dash_is_not_invented(self):
        result = self.step(paint(), [wheel()], 0)
        self.assertFalse(result["states"][0]["violated"])
        self.assertEqual(result["states"][0]["phase"], "unknown")

    def test_real_gap_is_not_joined_even_if_input_instance_contains_both_dashes(self):
        full = paint((20, 60), (140, 185))
        self.step(full, [], 0)
        result = self.step(full, [wheel((100, 95))], 0.1)
        self.assertFalse(result["states"][0]["violated"])
        self.assertEqual(len(result["lanes"]), 2)

    def test_observed_internal_hole_is_not_filled(self):
        mask = np.zeros(SHAPE, np.uint8)
        cv2.rectangle(mask, (40, 70), (160, 120), 255, -1)
        cv2.rectangle(mask, (80, 80), (120, 110), 0, -1)
        result = self.step(mask, [wheel()], 0)
        self.assertFalse(result["states"][0]["violated"])

    def test_expired_memory_is_unknown_not_a_false_exit(self):
        self.step(paint((50, 105)), [], 0)
        self.step(paint(), [wheel()], 0.1)
        result = self.step(paint(), [wheel()], 0.31)
        self.assertEqual(result["states"][0]["phase"], "unknown")
        self.assertFalse(result["states"][0]["confirmed"])
        self.assertFalse(any(e["event"] == "exited" for e in result["events"]))

    def test_registration_failure_does_not_reuse_stale_paint_or_claim_exit(self):
        self.step(paint((50, 105)), [], 0)
        self.step(paint(), [wheel()], 0.1)
        result = self.step(paint(), [wheel()], 0.2, Motion(reason="failed"))
        self.assertEqual(result["recovered_pixels"], 0)
        self.assertEqual(result["states"][0]["phase"], "unknown")
        self.assertFalse(any(e["event"] == "exited" for e in result["events"]))

    def test_short_registration_failure_can_recover_from_last_valid_reference(self):
        self.step(paint((50, 105)), [], 0)
        self.step(paint(), [wheel()], 0.1)
        self.step(paint(), [wheel()], 0.15, Motion(reason="failed"))
        result = self.step(paint(), [wheel()], 0.2)
        self.assertTrue(result["states"][0]["violated"])
        self.assertFalse(any(e["event"] == "entered" for e in result["events"]))

    def test_failed_registration_time_is_included_in_memory_expiration(self):
        self.step(paint((50, 105)), [], 0)
        self.step(paint(), [wheel()], 0.1)
        self.step(paint(), [wheel()], 0.25, Motion(reason="failed"))
        result = self.step(paint(), [wheel()], 0.4)
        self.assertFalse(result["states"][0]["violated"])
        self.assertEqual(result["states"][0]["phase"], "unknown")

    def test_long_unobservable_contact_does_not_silently_become_normal(self):
        full = paint((50, 105), (180, 220))
        self.step(full, [], 0)
        self.step(full, [wheel()], 0.1)
        events = []
        for tick in range(2, 17):
            result = self.step(paint((180, 220)), [wheel()], tick / 10)
            events.extend(result["events"])
        self.assertTrue(any(e["event"] == "lost" for e in events))
        self.assertFalse(any(e["event"] == "exited" for e in events))
        self.assertEqual(result["states"][0]["phase"], "unknown")

    def test_actual_exit_from_finite_dash_is_emitted(self):
        full = paint((50, 105))
        self.step(full, [], 0)
        self.step(full, [wheel()], 0.1)
        first = self.step(full, [wheel((120, 95))], 0.2)
        self.assertEqual(first["states"][0]["phase"], "exiting")
        result = self.step(full, [wheel((120, 95))], 0.26)
        self.assertTrue(result["states"][0]["exited"])
        self.assertFalse(result["states"][0]["confirmed"])

    def test_camera_translation_moves_memory_before_contact_test(self):
        self.step(paint((50, 105)), [], 0)
        matrix = np.array([[1, 0, 30], [0, 1, 0], [0, 0, 1]], float)
        result = self.step(paint(), [wheel((130, 95), (110, 65, 155, 115))],
                           0.1, Motion(matrix, True, 0.1, "translation"))
        self.assertTrue(result["states"][0]["violated"])

    def test_repeated_occlusion_predictions_do_not_refresh_memory_age(self):
        self.step(paint((50, 105)), [], 0)
        for t in [0.1, 0.2]:
            self.assertTrue(self.step(paint(), [wheel()], t)["states"][0]["violated"])
        result = self.step(paint(), [wheel()], 0.4)
        self.assertFalse(result["states"][0]["violated"])

    def test_large_timestamp_gap_invalidates_memory(self):
        self.step(paint((50, 105)), [], 0)
        result = self.engine.update(FRAME, [], [wheel()], 2.0)
        self.assertEqual(result["recovered_pixels"], 0)

    def test_reset_does_not_leak_previous_video(self):
        self.step(paint((50, 105)), [], 0)
        self.engine.reset()
        result = self.step(paint(), [wheel()], 0)
        self.assertFalse(result["states"][0]["violated"])

    def test_transition_to_a_different_dash_requires_exit_event(self):
        full = paint((40, 65), (140, 165))
        box = (20, 60, 190, 120)
        self.step(full, [], 0)
        first = self.step(full, [wheel((50, 95), box)], 0.1)
        old_id = first["states"][0]["lane_id"]
        handoff = self.step(full, [wheel((150, 95), box)], 0.2)
        self.assertTrue(handoff["states"][0]["violated"])
        self.assertNotEqual(handoff["states"][0]["hit_lane_id"], old_id)
        result = self.step(full, [wheel((150, 95), box)], 0.26)
        self.assertTrue(any(e["event"] == "exited" and e["lane_id"] == old_id for e in result["events"]))
        next_dash = self.step(full, [wheel((150, 95), box)], 0.3)
        self.assertTrue(any(e["event"] == "entered" for e in next_dash["events"]))
        self.assertNotEqual(next_dash["states"][0]["lane_id"], old_id)


class TimingTests(unittest.TestCase):
    def test_entry_uses_elapsed_video_time(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0.2))
        for frame in range(6):
            self.assertFalse(episode.update("inside", frame / 30, 1)["confirmed"])
        self.assertTrue(episode.update("inside", 6 / 30, 1)["confirmed"])

    def test_unknown_breaks_tentative_entry(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0.2))
        episode.update("inside", 0, 1)
        episode.update("unknown", 0.1)
        self.assertFalse(episode.update("inside", 0.2, 1)["confirmed"])
        self.assertTrue(episode.update("inside", 0.4, 1)["confirmed"])

    def test_unknown_is_not_counted_as_exit_time(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0, exit_hold_sec=0.1))
        episode.update("inside", 0, 1)
        episode.update("outside", 0.1)
        episode.update("unknown", 0.15)
        self.assertFalse(episode.update("outside", 0.2)["exited"])
        self.assertTrue(episode.update("outside", 0.3)["exited"])

    def test_short_contact_still_has_exit_without_becoming_confirmed_violation(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0.2, exit_hold_sec=0.05))
        initial = episode.update("inside", 0, 1)
        self.assertTrue(any(e["event"] == "contact_started" for e in initial["events"]))
        episode.update("outside", 0.04)
        last = episode.update("outside", 0.1)
        self.assertTrue(last["exited"])
        self.assertFalse(last["confirmed"])
        self.assertTrue(any(e["event"] == "contact_ended" for e in last["events"]))
        self.assertFalse(any(e["event"] == "exited" for e in last["events"]))

    def test_unconfirmed_occluded_contact_is_not_assumed_to_have_exited(self):
        engine = DashReferee(DashConfig(entry_hold_sec=0.2, legacy_contact_events=True))
        engine.update(FRAME, lanes(paint((50, 105))), [], 0, motion_override=IDENTITY)
        engine.update(FRAME, lanes(paint()), [wheel()], 0.1, motion_override=IDENTITY)
        result = engine.update(FRAME, [], [wheel()], 0.2, motion_override=Motion(reason="failed"))
        self.assertEqual(result["states"][0]["phase"], "unknown")
        self.assertFalse(result["states"][0]["exited"])

    def test_entry_time_does_not_transfer_between_dashes(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0.2))
        episode.update("inside", 0, 1)
        self.assertFalse(episode.update("inside", 0.15, 2)["confirmed"])
        self.assertFalse(episode.update("inside", 0.25, 2)["confirmed"])

    def test_reacquisition_requires_fresh_confirmation_without_duplicate_entry(self):
        episode = ContactEpisode(DashConfig(entry_hold_sec=0.1))
        episode.update("inside", 0, 1)
        self.assertTrue(episode.update("inside", 0.1, 1)["confirmed"])
        episode.update("unknown", 0.15)
        first = episode.update("inside", 0.2, 1)
        self.assertFalse(first["confirmed"])
        self.assertEqual(first["phase"], "recovering")
        last = episode.update("inside", 0.3, 1)
        self.assertTrue(last["confirmed"])
        self.assertFalse(any(e["event"] == "entered" for e in last["events"]))


class IdentityTests(unittest.TestCase):
    def test_detection_order_changes_keep_physical_wheel_ids(self):
        tracker = WheelAssociation(DashConfig())
        front = wheel((50, 95), (30, 65, 70, 115))
        rear = wheel((180, 95), (160, 65, 200, 115))
        first, _ = tracker.update([front, rear], 0, IDENTITY)
        second, _ = tracker.update([rear, front], 0.1, IDENTITY)
        self.assertEqual(first[0]["track_id"], second[1]["track_id"])
        self.assertEqual(first[1]["track_id"], second[0]["track_id"])

    def test_long_missing_wheel_gets_new_identity(self):
        tracker = WheelAssociation(DashConfig())
        first, _ = tracker.update([wheel()], 0, IDENTITY)
        second, expired = tracker.update([wheel()], 0.5, IDENTITY)
        self.assertNotEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertIn(first[0]["track_id"], expired)


class IntegrationTests(unittest.TestCase):
    def test_both_production_entrypoints_use_shared_engine_without_loading_models(self):
        import main
        import main_capture_dual_youtube
        for module in [main, main_capture_dual_youtube]:
            state = SimpleNamespace(paused=False)
            detector = SimpleNamespace(predict=lambda frame: [wheel()])
            with patch.object(module, "build_lane_data", return_value=lanes(paint((50, 105)))):
                image, states = module.process_frame(None, detector, [], FRAME, state, timestamp_sec=0)
            self.assertEqual(image.shape, FRAME.shape)
            self.assertNotIn("phase", states[0])
            self.assertTrue(states[0]["violated"])
            self.assertTrue(hasattr(state, "dash_referee"))

    def test_original_frame_only_mode_remains_available(self):
        import main
        import main_capture_dual_youtube
        from violation_tracker import ViolationTracker
        for module in [main, main_capture_dual_youtube]:
            state = SimpleNamespace(paused=False)
            detector = SimpleNamespace(predict=lambda frame: [wheel()])
            original = lanes(paint((50, 105)))[0]
            original.update(centerline=((50, 95), (105, 95)), half_width=5)
            trackers = [ViolationTracker(i) for i in range(module.MAX_WHEELS)]
            with patch.object(module, "TEMPORAL_DASH_ENABLED", False), patch.object(
                    module, "build_lane_data", return_value=[original]):
                _, states = module.process_frame(None, detector, trackers, FRAME, state, timestamp_sec=0)
            self.assertTrue(states[0]["violated"])
            self.assertFalse(hasattr(state, "dash_referee"))

    def test_new_run_reset_clears_contacts_but_keeps_observed_road(self):
        from dash_referee import process_temporal
        state = SimpleNamespace(contact_reset_requested=False)
        process_temporal(state, FRAME, lanes(paint((50, 105))), [wheel()], 0, 0)
        old_id = next(iter(state.dash_referee.wheels.tracks))
        state.contact_reset_requested = True
        process_temporal(state, FRAME, lanes(paint((50, 105))), [wheel()], 0, 0.1)
        self.assertNotIn(old_id, state.dash_referee.wheels.tracks)
        self.assertFalse(state.contact_reset_requested)


class RoadMotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.frame = rng.integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
        self.mask = np.full(SHAPE, 255, np.uint8)

    def translated(self, dx):
        matrix = np.array([[1, 0, dx], [0, 1, 1], [0, 0, 1]], float)
        return cv2.warpPerspective(self.frame, matrix, (SHAPE[1], SHAPE[0]))

    def test_estimated_motion_matches_known_translation(self):
        motion = RoadMotion(DashConfig())
        motion.estimate(self.frame, self.mask, [], 0)
        result = motion.estimate(self.translated(4), self.mask, [], 0.1)
        self.assertTrue(result.valid, result.reason)
        from dash_referee import transform_points
        predicted = transform_points([[130, 120]], result.matrix)[0]
        np.testing.assert_allclose(predicted, [134, 121], atol=0.8)

    def test_flat_images_do_not_produce_a_trusted_transform(self):
        motion = RoadMotion(DashConfig())
        motion.estimate(FRAME, self.mask, [], 0)
        self.assertFalse(motion.estimate(FRAME, self.mask, [], 0.1).valid)

    def test_blurred_failed_frame_does_not_replace_last_good_reference(self):
        motion = RoadMotion(DashConfig())
        motion.estimate(self.frame, self.mask, [], 0)
        self.assertFalse(motion.estimate(FRAME, self.mask, [], 0.1).valid)
        result = motion.estimate(self.translated(8), self.mask, [], 0.1)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.elapsed_sec, 0.2)

    def test_paint_crosscheck_accepts_correct_alignment_and_rejects_wrong_one(self):
        original = np.zeros(SHAPE, np.uint8)
        cv2.rectangle(original, (20, 45), (100, 55), 255, -1)
        cv2.rectangle(original, (140, 100), (230, 110), 255, -1)
        matrix = np.array([[1, 0, 4], [0, 1, 2], [0, 0, 1]], float)
        current = cv2.warpPerspective(original, matrix, (SHAPE[1], SHAPE[0]))
        excluded = np.zeros(SHAPE, np.uint8)
        self.assertTrue(RoadMotion.paint_support(original, current, matrix, excluded))
        wrong = matrix.copy(); wrong[0, 2] += 30
        self.assertFalse(RoadMotion.paint_support(original, current, wrong, excluded))

    def test_paint_crosscheck_does_not_use_occluded_paint_as_evidence(self):
        excluded = np.full(SHAPE, 255, np.uint8)
        self.assertFalse(RoadMotion.paint_support(self.mask, self.mask, np.eye(3), excluded))


if __name__ == "__main__":
    unittest.main()
