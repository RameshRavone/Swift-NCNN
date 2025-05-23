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

    init(withGPU gpu: Int32? = nil) {
        net = ncnn_net_create()
        option = ncnn_option_create()

        let useGPU = gpu != nil
        assert(!useGPU || (gpu! >= 0))

        ncnn_option_set_num_threads(option, useGPU ? 4 : 2)
        ncnn_option_set_use_vulkan_compute(option, useGPU ? 1 : 0)

        ncnn_net_set_option(net, option)

        if useGPU {
            ncnn_net_set_vulkan_device(net, gpu!)
        }
    }

    public func Release(mat: ncnn_mat_t?) {
        if let mat {
            ncnn_mat_destroy(mat)
        }
    }

    public mutating func Release() {
        if let net, let option {
            ncnn_net_destroy(net)
            ncnn_option_destroy(option)

            self.net = nil
            self.option = nil
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

    public func CreateInput(data: UnsafeMutableRawPointer, shape: [Int64]) throws -> ncnn_mat_t {
        assert(net != nil)

        let width = Int32(shape[2])
        let height = Int32(shape[3])
        let channels = Int32(shape[1])
        let count = Int(shape[0] * shape[1] * shape[2] * shape[3])

        switch channels {
        case 2:
            return ncnn_mat_create_external_2d(width, height, data, nil)
        case 3:
            if let mat = ncnn_mat_create_3d(width, height, channels, nil) {
                let _ptr = ncnn_mat_get_data(mat)
                _ptr?.copyMemory(from: data, byteCount: count * MemoryLayout<Float>.stride)
                return mat
            }
            throw "Failed to create Input Matrix/Tensor"
        default: throw ("Wrong channel count")
        }
    }

    public func Run(
        withInputs inputs: [String: ncnn_mat_t],
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

        var result: [String: ncnn_mat_t] = [:]
        for name in outputNames {
            var out: ncnn_mat_t?
            ncnn_extractor_extract(ex, name, &out)

            if let out {
                result[name] = out
            }
        }
        assert(outputNames.count == result.count)

        ncnn_extractor_destroy(ex)
        return result
    }

    public func GetData<DataType: Copyable>(
        from mat: ncnn_mat_t,
        into data: inout [DataType],
        size: Int
    ) throws {
        assert(net != nil)

        let read: UnsafeMutableRawPointer? = ncnn_mat_get_data(mat)
        assert(read != nil)

        if let outdata = read?.assumingMemoryBound(to: DataType.self) {
            data = Array(UnsafeBufferPointer(start: outdata, count: size))
        } else {
            throw "Failed to type conversion!"
        }
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

public func MakeNCNN(withGPU gpu: Int32?) -> NCNN {
    return NCNN(withGPU: gpu)
}
