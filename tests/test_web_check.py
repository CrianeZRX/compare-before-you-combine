"""The browser check on the project page (docs/assets/check.js) gives the same result as compare6d check."""

import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from compare6d.check import check, load_table

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "assets" / "check.js"
NODE = shutil.which("node")
RUN = ("const C = require(process.argv[1]), fs = require('fs');"
       "process.stdout.write(JSON.stringify(C.check(C.parseCSV(fs.readFileSync(process.argv[2], 'utf8')))));")


@unittest.skipUnless(NODE and SCRIPT.exists(), "needs node and docs/assets/check.js")
class BrowserCheckMatchesPython(unittest.TestCase):
    def browser(self, path):
        done = subprocess.run([NODE, "-e", RUN, str(SCRIPT), str(path)], capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    def python(self, path):
        return json.loads(json.dumps(check(load_table(path))))

    def test_same_result(self):
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            tables = [ROOT / "examples" / "scores_table.csv"]
            # 4 images: too few; 9: exact sign-flip test; 20 and 40: Monte Carlo sign-flip test
            for images, keep in ((4, 0.9), (9, 0.8), (20, 0.6), (40, 0.5)):
                lines = ["scene_id,im_id,obj_id,score,correct"]
                for image in range(images):
                    for obj in (1, 2):
                        for _ in range(rng.randint(1, 4)):
                            correct = rng.random() < 0.5
                            boost = 0.3 if correct != (rng.random() < keep) else 0.0
                            score = 0.5 if rng.random() < 0.1 else round(rng.random() + boost, 3)
                            lines.append(f"{image % 3},{image},{obj},{score},{int(correct)}")
                path = Path(tmp) / f"table_{images}.csv"
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                tables.append(path)
            readings = set()
            for path in tables:
                with self.subTest(table=path.name):
                    expected = self.python(path)
                    self.assertEqual(self.browser(path), expected)
                    readings.add(expected["reading"])
            self.assertIn("reversed", readings)
            self.assertIn("too_few_images", readings)


if __name__ == "__main__":
    unittest.main()
