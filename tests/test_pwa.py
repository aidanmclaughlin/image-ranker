import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class PwaTests(unittest.TestCase):
    def test_manifest_is_installable_and_local_only(self):
        manifest = json.loads((WEB / "manifest.webmanifest").read_text())
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["start_url"], "/#rank")
        self.assertEqual(manifest["scope"], "/")
        self.assertTrue(any(icon["purpose"] == "maskable" for icon in manifest["icons"]))
        png_sizes = {
            icon.get("sizes") for icon in manifest["icons"] if icon.get("type") == "image/png"
        }
        self.assertTrue({"192x192", "512x512"}.issubset(png_sizes))
        self.assertNotIn("screenshots", manifest)

        for icon in manifest["icons"]:
            path = WEB / icon["src"].removeprefix("/")
            self.assertTrue(path.is_file())
            if path.suffix == ".svg":
                ET.parse(path)

        index = (WEB / "index.html").read_text()
        self.assertIn('rel="apple-touch-icon"', index)
        self.assertTrue((WEB / "icons" / "apple-touch-icon.png").is_file())

    def test_service_worker_caches_shell_but_not_private_data(self):
        worker = (WEB / "service-worker.js").read_text()
        shell_section = worker.split("const SHELL_FILES = [", 1)[1].split("];", 1)[0]
        for private_prefix in ("/api/", "/media/", "/thumb/"):
            self.assertNotIn(private_prefix, shell_section)
        self.assertIn("PRIVATE_PATH.test(url.pathname)", worker)
        self.assertIn('caches.match("/index.html")', worker)

    def test_vercel_deployment_is_full_stack_nextjs(self):
        config = json.loads((ROOT / "vercel.json").read_text())
        self.assertEqual(config["framework"], "nextjs")

        ignored = (ROOT / ".vercelignore").read_text()
        for private_path in ("data/", "models/", "*.sqlite3", ".env"):
            self.assertIn(private_path, ignored)

    def test_remote_connection_uses_fragment_and_authenticated_blobs(self):
        html = (WEB / "index.html").read_text()
        app = (WEB / "app.js").read_text()
        self.assertIn('rel="manifest"', html)
        self.assertIn('id="connection-dialog"', html)
        self.assertIn('window.location.hash.startsWith("#connect?")', app)
        self.assertIn("window.history.replaceState", app)
        self.assertIn('headers.set("Authorization", `Bearer ${state.apiToken}`)', app)
        self.assertIn("URL.createObjectURL(blob)", app)
        self.assertIn("URL.revokeObjectURL", app)


if __name__ == "__main__":
    unittest.main()
