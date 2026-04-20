load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")

package(
    default_applicable_licenses = ["//third_party/py/google/adk:package_license"],
    default_visibility = ["//visibility:private"],
)

pytest_test(
    name = "test_local_environment",
    srcs = ["test_local_environment.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    deps = [
        "//third_party/py/google/adk",
        "//third_party/py/pytest_asyncio",
    ],
)

pytest_test(
    name = "test_skill_toolset",
    srcs = ["test_skill_toolset.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    deps = [
        "//third_party/py/google/adk",
        "//third_party/py/google/genai",
        "//third_party/py/pytest_asyncio",
    ],
)

pytest_test(
    name = "test_context_cancellation",
    srcs = ["test_context_cancellation.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    deps = [
        "//third_party/py/google/adk",
        "//third_party/py/google/genai",
        "//third_party/py/pytest_asyncio",
    ],
)
