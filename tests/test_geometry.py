# Handles test geometry behavior.
import unittest

from tools.components.geometry import ground_to_radar_screen, image_to_relative_ground


class TestGeometryMappings(unittest.TestCase):
    def test_image_to_relative_ground_center_maps_to_origin_x(self) -> None:
        # Performs test image to relative ground center maps to origin x logic.
        x, z = image_to_relative_ground(
            pixel_x=320,
            pixel_y=240,
            frame_width=640,
            frame_height=480,
            max_ground_x=10.0,
            max_ground_z=20.0,
        )
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(z, 10.0, places=6)

    def test_image_to_relative_ground_left_and_right_edges(self) -> None:
        # Performs test image to relative ground left and right edges logic.
        left_x, _ = image_to_relative_ground(0, 480, 640, 480)
        right_x, _ = image_to_relative_ground(640, 480, 640, 480)

        self.assertAlmostEqual(left_x, -10.0, places=6)
        self.assertAlmostEqual(right_x, 10.0, places=6)

    def test_ground_to_radar_screen_center_and_camera_edge(self) -> None:
        # Performs test ground to radar screen center and camera edge logic.
        panel_w, panel_h = 680, 600
        center_x, camera_y = ground_to_radar_screen(
            ground_x=0.0,
            ground_z=0.0,
            panel_width=panel_w,
            panel_height=panel_h,
            max_ground_x=10.0,
            max_ground_z=20.0,
            margin=40,
        )

        self.assertEqual(center_x, panel_w // 2)
        self.assertEqual(camera_y, panel_h - 40)

    def test_ground_to_radar_screen_farther_z_moves_up(self) -> None:
        # Performs test ground to radar screen farther z moves up logic.
        near_point = ground_to_radar_screen(0.0, 1.0, 680, 600)
        far_point = ground_to_radar_screen(0.0, 18.0, 680, 600)

        self.assertLess(far_point[1], near_point[1])

    def test_image_to_relative_ground_clamps_out_of_frame_pixels(self) -> None:
        # Performs test image to relative ground clamps out of frame pixels logic.
        x_low, z_low = image_to_relative_ground(-100.0, 9999.0, 640, 480)
        x_high, z_high = image_to_relative_ground(9999.0, -100.0, 640, 480)

        self.assertAlmostEqual(x_low, -10.0, places=6)
        self.assertAlmostEqual(z_low, 0.0, places=6)
        self.assertAlmostEqual(x_high, 10.0, places=6)
        self.assertAlmostEqual(z_high, 20.0, places=6)

    def test_geometry_mappings_require_positive_ground_extents(self) -> None:
        # Performs test geometry mappings require positive ground extents logic.
        with self.assertRaises(ValueError):
            image_to_relative_ground(0.0, 0.0, 640, 480, max_ground_x=0.0, max_ground_z=20.0)
        with self.assertRaises(ValueError):
            image_to_relative_ground(0.0, 0.0, 640, 480, max_ground_x=10.0, max_ground_z=0.0)
        with self.assertRaises(ValueError):
            ground_to_radar_screen(0.0, 0.0, 680, 600, max_ground_x=0.0, max_ground_z=20.0)
        with self.assertRaises(ValueError):
            ground_to_radar_screen(0.0, 0.0, 680, 600, max_ground_x=10.0, max_ground_z=0.0)

    def test_ground_to_radar_screen_clamps_to_panel_margin(self) -> None:
        # Performs test ground to radar screen clamps to panel margin logic.
        panel_w, panel_h, margin = 680, 600, 40
        left_bottom = ground_to_radar_screen(-999.0, -999.0, panel_w, panel_h, margin=margin)
        right_top = ground_to_radar_screen(999.0, 999.0, panel_w, panel_h, margin=margin)

        self.assertEqual(left_bottom[0], margin)
        self.assertEqual(left_bottom[1], panel_h - margin)
        self.assertEqual(right_top[0], panel_w - margin)
        self.assertEqual(right_top[1], margin)


if __name__ == "__main__":
    unittest.main()
