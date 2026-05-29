load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")

package(
    default_applicable_licenses = ["//third_party/py/google/adk:package_license"],
    default_visibility = ["//visibility:private"],
)

pytest_test(
    name = "test_auto_tracing_plugin",
    srcs = ["test_auto_tracing_plugin.py"],
    deps = [
        "//third_party/py/google/adk",
        "//third_party/py/opentelemetry:opentelemetry_api",
        "//third_party/py/opentelemetry:opentelemetry_sdk",
    ],
)
