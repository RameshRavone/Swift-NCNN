// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package: Package = .init(
    name: "NCNN",
    products: [
        .library(name: "NCNN", targets: ["NCNN"]),
        .library(name: "CNCNN", targets: ["CNCNN"])
    ],
    targets: [
        .systemLibrary(name: "CNCNN", pkgConfig: "libncnn"),
        .target(
            name: "NCNN",
            dependencies: ["CNCNN"]
        )
    ]
)
