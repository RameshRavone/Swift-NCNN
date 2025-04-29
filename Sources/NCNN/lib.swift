// lib.swift
// Copyright (c) 2025 FrogSquare
// Created by Ramesh (Ravone)

import CNCNN
import Foundation

extension Swift.String: Swift.Error {}

@frozen
public struct NCNN {
    var net: ncnn_net_t?
    var option: ncnn_option_t?

    public var isValid: Bool {
        net != nil && option != nil
    }

    init() {
        net = ncnn_net_create()
        option = ncnn_option_create()

        ncnn_option_set_num_threads(option, 4)
        ncnn_option_set_use_vulkan_compute(option, 1)

        ncnn_net_set_option(net, option)
    }

    public func Release(mat: ncnn_mat_t?) {
        if let mat {
            ncnn_mat_destroy(mat)
        }
    }

    public func Release() {
        if let net, let option {
            ncnn_net_destroy(net)
            ncnn_option_destroy(option)
        }
    }

    public func CreateSession(param ppath: String, model mpath: String) throws {
        guard let net else {
            throw "Failed to create NCNN interface"
        }

        if (ncnn_net_load_param(net, ppath) != 0) ||
            (ncnn_net_load_model(net, mpath) != 0)
        {
            throw "Failed to load model"
        }
    }

    public func CreateInput(data: [Float], shape: [Int64]) throws -> ncnn_mat_t? {
        assert(net != nil)

        let width = Int32(shape[2])
        let height = Int32(shape[3])
        let channels = Int32(shape[1])
        let count = data.count

        var data = data

        return try data.withUnsafeMutableBytes { ptr in

            switch channels {
            case 2:
                return ncnn_mat_create_external_2d(width, height, ptr.baseAddress, nil)
            case 3:
                let mat = ncnn_mat_create_3d(width, height, channels, nil)
                let _ptr = ncnn_mat_get_data(mat)
                _ptr?.copyMemory(from: ptr.baseAddress!, byteCount: count * MemoryLayout<Float>.stride)
                return mat
            default: throw ("Wrong channel count")
            }
        }
    }

    public func Run(
        withInput inputs: [String: ncnn_mat_t?],
        outputNames: [String]
    ) throws -> [String: ncnn_mat_t] {
        assert(!inputs.isEmpty && !outputNames.isEmpty)

        guard let net else {
            throw "Failed to create NCNN interface"
        }

        let ex = ncnn_extractor_create(net)

        for input in inputs {
            ncnn_extractor_input(ex, input.key, input.value)
        }

        var out: [ncnn_mat_t?] = Array(repeating: nil, count: outputNames.count)

        outputNames.withUnsafeBytes { names in
            _ = ncnn_extractor_extract(
                ex,
                names.baseAddress,
                &out
            )
        }
        assert(outputNames.count == out.count)

        var result: [String: ncnn_mat_t] = [:]
        for (i, o) in out.enumerated() {
            if let o {
                result[outputNames[i]] = o
            }
        }

        inputs.forEach {
            ncnn_mat_destroy($1)
        }

        ncnn_extractor_destroy(ex)
        return result
    }

    public func GetData(
        from mat: ncnn_mat_t,
        into data: inout [Float],
        _ width: inout Int32,
        _ height: inout Int32,
        _ channels: inout Int32
    ) throws {
        assert(net != nil)

        width = ncnn_mat_get_w(mat)
        height = ncnn_mat_get_h(mat)
        channels = ncnn_mat_get_c(mat)

        let read: UnsafeMutableRawPointer? = ncnn_mat_get_data(mat)
        data = if let outdata = read?.assumingMemoryBound(to: Float.self) {
            Array(UnsafeBufferPointer(start: outdata, count: Int(width * height * channels)))
        } else {
            []
        }
    }
}

public func MakeNCNN() -> NCNN {
    return NCNN()
}
