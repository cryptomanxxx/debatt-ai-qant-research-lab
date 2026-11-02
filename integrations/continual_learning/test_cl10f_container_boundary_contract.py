"""CL-10F mutation tests; verify lists only, never call Docker."""
import unittest

from cl10f_container_boundary_contract import (
    FIXED_FLAGS, validate_review_only_container_command,
)

IMAGE = "busybox@sha256:fd7dc98638c8e305f4dc34e979f1c0fdfdcaeb0fbf8fcff77ae834b6da3d7e6e"


def candidate():
    return ["docker", "run", *FIXED_FLAGS, IMAGE]


class TestCL10F(unittest.TestCase):
    def test_exact_review_only_declaration(self):
        result = validate_review_only_container_command(candidate())
        self.assertEqual(result["network"], "none")
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["runtime_boundary_verified"])
        self.assertFalse(result["training_authorized"])

    def test_every_fixed_flag_is_required(self):
        for index in range(2, len(candidate()) - 1):
            argv = candidate()
            argv.pop(index)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_review_only_container_command(argv)

    def test_unsafe_mutations_rejected(self):
        for replacement in (
            "--network=bridge", "--network=host", "--privileged",
            "--cap-add=SYS_ADMIN", "--read-write", "--user=0:0",
            "--memory=unlimited", "--pull=always",
            "--security-opt=seccomp=unconfined",
        ):
            argv = candidate()
            argv[4] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_review_only_container_command(argv)
        for extra in ("python", "--env=SECRET=secret", "-v", "--privileged"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_review_only_container_command(candidate() + [extra])

    def test_no_mutable_or_malformed_images(self):
        for image in ("busybox:latest", "busybox@sha256:abc", "", 42):
            argv = candidate()
            argv[-1] = image
            with self.subTest(image=image), self.assertRaises(ValueError):
                validate_review_only_container_command(argv)


if __name__ == "__main__":
    unittest.main()
