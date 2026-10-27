import unittest

from tools.components.geometry import ground_to_radar_screen, image_to_relative_ground


class TestGeometryMappings(unittest.TestCase):
    def test_image_to_relative_ground_center_maps_to_origin_x(self) -> None:
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
        left_x, _ = image_to_relative_ground(0, 480, 640, 480)
        right_x, _ = image_to_relative_ground(640, 480, 640, 480)

        self.assertAlmostEqual(left_x, -10.0, places=6)
        self.assertAlmostEqual(right_x, 10.0, places=6)

    def test_ground_to_radar_screen_center_and_camera_edge(self) -> None:
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
        near_point = ground_to_radar_screen(0.0, 1.0, 680, 600)
        far_point = ground_to_radar_screen(0.0, 18.0, 680, 600)

        self.assertLess(far_point[1], near_point[1])


if __name__ == "__main__":
    unittest.main()
