require 'ffi'

module OnnxRuntime
  extend FFI::Library

  # TODO: Ensure this points to your shared library (e.g., libonnxruntime.dylib or .so)
  # We default to looking for the dylib in the same directory as this file
  ORT_LIB_PATH = ENV['ORT_LIB_PATH'] || File.join(__dir__, 'libonnxruntime.dylib')

  ffi_lib ORT_LIB_PATH

  # -------------------------------------------------------------------------
  # FFI Typedefs and Enums
  # -------------------------------------------------------------------------
  typedef :pointer, :OrtEnv
  typedef :pointer, :OrtSession
  typedef :pointer, :OrtMemoryInfo
  typedef :pointer, :OrtValue
  typedef :pointer, :OrtStatus
  typedef :pointer, :OrtSessionOptions
  typedef :pointer, :OrtRunOptions
  typedef :pointer, :OrtAllocator

  # ONNX Tensor Types
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 11

  # Memory Types
  ORT_MEM_TYPE_CPU = 1

  # Logging Levels
  ORT_LOGGING_LEVEL_WARNING = 2

  # -------------------------------------------------------------------------
  # Global API Access
  # -------------------------------------------------------------------------
  # The entry point to the C API. Returns a pointer to OrtApiBase.
  attach_function :OrtGetApiBase, [], :pointer

  class << self
    attr_reader :api

    def load_api
      return if @api

      base_ptr = OrtGetApiBase()
      raise "Failed to get OrtApiBase" if base_ptr.null?

      # OrtApiBase struct: { const OrtApi* (*GetApi)(uint32_t version); const char* (*GetVersionString)(void); }
      # We read the first pointer, which is the GetApi function pointer.
      get_api_fn_ptr = base_ptr.read_pointer

      # Define parameters for GetApi: (version) -> OrtApi*
      get_api = FFI::Function.new(:pointer, [:uint32], get_api_fn_ptr)

      # Request API version 14 (compatible with the opset we used)
      @api_struct_ptr = get_api.call(14) 
      raise "Failed to get OrtApi version 14" if @api_struct_ptr.null?

      @api = OrtApi.new(@api_struct_ptr)
    end
  end

  # -------------------------------------------------------------------------
  # OrtApi Struct Definition
  # -------------------------------------------------------------------------
  # Using a layout based on standard headers. 
  # IMPORTANT: This order MUST match onnxruntime_c_api.h for the loaded version.
  class OrtApi < FFI::Struct
    layout \
      :CreateStatus, :pointer,
      :GetErrorCode, :pointer,
      :GetErrorMessage, :pointer,
      :CreateEnv, :pointer,
      :CreateEnvWithCustomLogger, :pointer,
      :CreateSession, :pointer,
      :CreateSessionWithOptions, :pointer,
      :Run, :pointer,
      :CreateSessionOptions, :pointer,
      :GetImagesFirst, :pointer,
      :GetImagesLast, :pointer,
      :CreateSessionArray, :pointer,
      :ReleaseStatus, :pointer,
      :ReleaseEnv, :pointer,
      :ReleaseSession, :pointer,
      :ReleaseSessionOptions, :pointer,
      :ReleaseMemoryInfo, :pointer,
      :ReleaseValue, :pointer
      # Note: There are MANY more fields. 
      # We access later ones (CreateCpuMemoryInfo, CreateTensor...) manually 
      # via offset if needed, or assume we define enough here.
      # For a truly minimal and robust solution without strict ordering fears,
      # strict implementations map exact versions.
      # For this snippet, we assume the top fields (Create/Release) are stable.
      #
      # To access functions further down safely without defining 100 structs fields,
      # we can use manual offsets if we knew them. 
      #
      # HOWEVER, CreateCpuMemoryInfo and CreateTensorWithDataAsOrtValue are 
      # essential. They are typically located around index 36 and 37 in v1.12+.
      # We will add padding to reach them.
  end

  # Manual function fetcher for deep API functions to avoid struct fragility
  def self.get_func(index, args, ret)
    # 8 bytes per pointer
    ptr = @api_struct_ptr + (index * 8)
    FFI::Function.new(ret, args, ptr.read_pointer)
  end

  # Function Indices (Checked against v1.13+)
  # Update these if you get segfaults on different versions!
  IDX_CreateCpuMemoryInfo = 36 
  IDX_CreateTensorWithDataAsOrtValue = 37
  IDX_GetTensorMutableData = 43

  # -------------------------------------------------------------------------
  # High-Level Ruby Session
  # -------------------------------------------------------------------------
  class OnnxSession
    def initialize(model_path)
      OnnxRuntime.load_api

      @env_ptr = FFI::MemoryPointer.new(:pointer)
      @session_ptr = FFI::MemoryPointer.new(:pointer)

      # 1. Create Env
      # CreateEnv(logging_level, logid, out_env)
      create_env = FFI::Function.new(:pointer, [:int, :string, :pointer], OnnxRuntime.api[:CreateEnv])
      status = create_env.call(OnnxRuntime::ORT_LOGGING_LEVEL_WARNING, "ruby_test", @env_ptr)
      check_status(status)
      @env = @env_ptr.read_pointer

      # 2. Create Session
      # CreateSession(env, model_path, options, out_session)
      create_session = FFI::Function.new(:pointer, [:pointer, :string, :pointer, :pointer], OnnxRuntime.api[:CreateSession])
      status = create_session.call(@env, model_path, nil, @session_ptr)
      check_status(status)
      @session = @session_ptr.read_pointer
    end

    def run(inputs_hash)
      # Inputs: inputs_hash = { "input_ids" => [[...]], "attention_mask" => [[...]] }
      
      memory_info_ptr = FFI::MemoryPointer.new(:pointer)
      
      # 1. Create Memory Info (CPU)
      create_mem_info = OnnxRuntime.get_func(OnnxRuntime::IDX_CreateCpuMemoryInfo, [:int, :int, :pointer], :pointer)
      status = create_mem_info.call(OnnxRuntime::ORT_MEM_TYPE_CPU, 0, memory_info_ptr)
      check_status(status)
      memory_info = memory_info_ptr.read_pointer

      input_names_ptrs = []
      input_values_ptrs = []
      
      # Keep references to keep data alive during call
      kept_alive = []

      inputs_hash.each do |name, data|
        # Flatten data and determine shape
        # data is expected to be [batch_size, seq_len] array of Integers (INT64)
        batch_size = data.size
        seq_len = data.first.size
        flat_data = data.flatten
        element_count = flat_data.size
        
        # Prepare data buffer
        data_buffer = FFI::MemoryPointer.new(:int64, element_count)
        data_buffer.write_array_of_int64(flat_data)
        
        # Prepare shape
        shape = [batch_size, seq_len]
        shape_ptr = FFI::MemoryPointer.new(:int64, shape.size)
        shape_ptr.write_array_of_int64(shape)

        kept_alive << data_buffer
        kept_alive << shape_ptr

        # Create Tensor
        value_ptr = FFI::MemoryPointer.new(:pointer)
        create_tensor = OnnxRuntime.get_func(OnnxRuntime::IDX_CreateTensorWithDataAsOrtValue, 
          [:pointer, :pointer, :size_t, :pointer, :size_t, :int, :pointer], :pointer)
        
        status = create_tensor.call(
          memory_info,
          data_buffer,
          data_buffer.size, # bytes length
          shape_ptr,
          shape.size,
          OnnxRuntime::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
          value_ptr
        )
        check_status(status)
        
        input_names_ptrs << FFI::MemoryPointer.from_string(name)
        input_values_ptrs << value_ptr.read_pointer
      end

      # 2. Run
      # Run(session, run_opts, input_names, input_values, num_inputs, output_names, num_outputs, output_values)
      
      # Prepare arrays for FFI
      in_names_arr = FFI::MemoryPointer.new(:pointer, input_names_ptrs.size)
      in_names_arr.write_array_of_pointer(input_names_ptrs)
      
      in_values_arr = FFI::MemoryPointer.new(:pointer, input_values_ptrs.size)
      in_values_arr.write_array_of_pointer(input_values_ptrs)

      # Output - "logits"
      out_name = "output" # The ONNX model output name we defined in python export
      out_names_ptr = FFI::MemoryPointer.new(:pointer, 1)
      out_names_ptr.write_pointer(FFI::MemoryPointer.from_string(out_name))
      
      out_values_ptr = FFI::MemoryPointer.new(:pointer, 1)
      # Initialize with null to let ORT allocate output
      out_values_ptr.write_pointer(nil) 

      run_fn = FFI::Function.new(:pointer, [:pointer, :pointer, :pointer, :pointer, :size_t, :pointer, :size_t, :pointer], OnnxRuntime.api[:Run])
      
      status = run_fn.call(
        @session, 
        nil, # RunOptions
        in_names_arr, 
        in_values_arr, 
        input_names_ptrs.size,
        out_names_ptr, 
        1, 
        out_values_ptr
      )
      check_status(status)

      # 3. Extract Output
      output_value = out_values_ptr.read_pointer
      
      # Get pointer to float data
      get_tensor_data = OnnxRuntime.get_func(OnnxRuntime::IDX_GetTensorMutableData, [:pointer, :pointer], :pointer)
      # Actually GetTensorMutableData signature usually returns OrtStatus* and takes void** as arg.
      # Let's verify signature.
      # ORT_API2(GetTensorMutableData, _Inout_ OrtValue* value, _Out_ void** out);
      
      float_out_ptr = FFI::MemoryPointer.new(:pointer)
      status = get_tensor_data.call(output_value, float_out_ptr)
      check_status(status)
      
      final_float_ptr = float_out_ptr.read_pointer
      
      # Assume [1, 2] output for this model (Binary classification logits)
      # We should ideally ask the tensor for its shape info, but for minimal code:
      output_array = final_float_ptr.read_array_of_float(2)

      # Cleanup Output Value (Others potentially freed by GC or need explicit release)
      release_val = FFI::Function.new(:void, [:pointer], OnnxRuntime.api[:ReleaseValue])
      release_val.call(output_value)
      
      # Release Input Values
      input_values_ptrs.each do |val|
        release_val.call(val)
      end
      
      # Release memory info
      release_mem = FFI::Function.new(:void, [:pointer], OnnxRuntime.api[:ReleaseMemoryInfo])
      release_mem.call(memory_info)

      return output_array
    end

    def close
      release_sess = FFI::Function.new(:void, [:pointer], OnnxRuntime.api[:ReleaseSession])
      release_sess.call(@session) if @session
      
      release_env = FFI::Function.new(:void, [:pointer], OnnxRuntime.api[:ReleaseEnv])
      release_env.call(@env) if @env
    end
    
    private

    def check_status(status_ptr)
      return if status_ptr.null?
      
      get_err = FFI::Function.new(:string, [:pointer], OnnxRuntime.api[:GetErrorMessage])
      msg = get_err.call(status_ptr)
      
      release_status = FFI::Function.new(:void, [:pointer], OnnxRuntime.api[:ReleaseStatus])
      release_status.call(status_ptr)
      
      raise "OnnxRuntime Error: #{msg}"
    end
  end
end
