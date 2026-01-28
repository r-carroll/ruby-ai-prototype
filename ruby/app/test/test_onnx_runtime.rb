# test/test_onnx_runtime.rb
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)

require "onnxruntime"

# Adjust model path if needed
MODEL_PATH = File.expand_path("../models/test_add.onnx", __dir__)


# Create an inference session using the Gem's API
model = OnnxRuntime::Model.new(MODEL_PATH)

# Our test model expects a float32 tensor [1, 3] called "input"
# The gem accepts a Hash { "input_name" => array_data }
input_data = [1.0, 2.0, 3.0]
outputs = model.predict({ "input" => input_data })

puts "Outputs: #{outputs.inspect}"
