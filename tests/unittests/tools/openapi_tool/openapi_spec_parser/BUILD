load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")

package(
    default_applicable_licenses = ["//third_party/py/google/adk:package_license"],
    default_visibility = ["//visibility:private"],
)

pytest_test(
    name = "test_openapi_spec_parser",
    srcs = ["test_openapi_spec_parser.py"],
    deps = [
        "//third_party/py/google/adk",
    ],
)

pytest_test(
    name = "test_openapi_toolset",
    srcs = ["test_openapi_toolset.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    data = [
        "test.yaml",
    ],
    deps = [
        "//third_party/py/fastapi",
        "//third_party/py/google/adk",
        "//third_party/py/pytest_asyncio",
        "//third_party/py/yaml",
    ],
)

pytest_test(
    name = "test_operation_parser",
    srcs = ["test_operation_parser.py"],
    deps = [
        "//third_party/py/fastapi",
        "//third_party/py/google/adk",
    ],
)

pytest_test(
    name = "test_rest_api_tool",
    srcs = ["test_rest_api_tool.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    deps = [
        "//third_party/py/fastapi",
        "//third_party/py/google/adk",
        "//third_party/py/httpx",
        "//third_party/py/pytest_asyncio",
        "//third_party/py/requests",
    ],
)

pytest_test(
    name = "test_tool_auth_handler",
    srcs = ["test_tool_auth_handler.py"],
    args = [
        "-p",
        "pytest_asyncio.plugin",
    ],
    deps = [
        "//third_party/py/google/adk",
        "//third_party/py/pytest_asyncio",
    ],
)
