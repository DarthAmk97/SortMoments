import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class FakeFaceModel:
    def get(self, image_rgb):
        from sortmoments import DetectedFace
        return [DetectedFace(bbox=[5, 5, 30, 30], det_score=0.99)]


class FakeEmbeddingModel:
    def embed(self, image_rgb, face):
        return np.ones(4, dtype=np.float32)


class FakeGroupingModel:
    def group(self, face_records, similarity_threshold=0.5):
        return {"custom_person": [record.face_path for record in face_records]}


class LibraryApiTests(unittest.TestCase):
    def test_organizer_class_accepts_custom_models(self):
        from sortmoments import OrganizationResult, SortMomentsConfig, SortMomentsOrganizer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "sample.jpg"
            image = np.full((64, 64, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (5, 5), (30, 30), (0, 0, 0), -1)
            cv2.imwrite(str(img_path), image)

            organizer = SortMomentsOrganizer(
                SortMomentsConfig(
                    min_face_size=10,
                    blur_threshold=0,
                    keep_intermediate=True,
                    batch_size=1,
                    max_workers=1,
                ),
                face_model=FakeFaceModel(),
                embedding_model=FakeEmbeddingModel(),
                grouping_model=FakeGroupingModel(),
            )
            result = organizer.organize(root)

            self.assertIsInstance(result, OrganizationResult)
            self.assertEqual(result.person_count, 1)
            self.assertTrue((result.output_folder / "custom_person" / "sample.jpg").exists())

    def test_organize_photos_accepts_custom_models(self):
        from sortmoments import SortMomentsConfig, organize_photos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "sample.jpg"
            image = np.full((64, 64, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (5, 5), (30, 30), (0, 0, 0), -1)
            cv2.imwrite(str(img_path), image)

            result = organize_photos(
                root,
                config=SortMomentsConfig(
                    min_face_size=10,
                    blur_threshold=0,
                    keep_intermediate=True,
                    batch_size=1,
                    max_workers=1,
                ),
                face_model=FakeFaceModel(),
                embedding_model=FakeEmbeddingModel(),
                grouping_model=FakeGroupingModel(),
            )

            self.assertEqual(result.person_count, 1)
            self.assertEqual(result.face_count, 1)
            self.assertTrue((result.output_folder / "custom_person" / "sample.jpg").exists())
            self.assertTrue(result.embeddings_file.exists())


if __name__ == "__main__":
    unittest.main()
