# config/initializers/model_loader.rb

require 'onnxruntime'
require 'tokenizers'

class ModelLoader
  include Singleton

  attr_reader :bert_session, :bert_tokenizer, :gpt_session, :gpt_tokenizer

  def initialize
    @models_path = Rails.root.join('vendor', 'models')
    
    # Load BERT Sentiment Model
    bert_path = @models_path.join('model.onnx')
    vocab_path = @models_path.join('vocab.txt')
    
    if File.exist?(bert_path) && File.exist?(vocab_path)
      @bert_session = OnnxRuntime::Model.new(bert_path.to_s)
      vocab = File.readlines(vocab_path).each_with_index.to_h { |line, i| [line.strip, i] }
      @bert_tokenizer = Tokenizers::Tokenizer.new(Tokenizers::Models::WordPiece.new(vocab: vocab))
      @bert_tokenizer.normalizer = Tokenizers::Normalizers::BertNormalizer.new
      @bert_tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::BertPreTokenizer.new
    end

    # Load GPT-2 Japanese Generative Model
    gpt_path = @models_path.join('gpt2_japanese', 'model.onnx')
    puts "DEBUG: Checking GPT path: #{gpt_path} (Exists: #{File.exist?(gpt_path)})"
    if File.exist?(gpt_path)
      @gpt_session = OnnxRuntime::Model.new(gpt_path.to_s)
      gpt_tokenizer_path = @models_path.join('gpt2_japanese', 'tokenizer.json')
      puts "DEBUG: Checking GPT Tokenizer path: #{gpt_tokenizer_path} (Exists: #{File.exist?(gpt_tokenizer_path)})"
      if File.exist?(gpt_tokenizer_path)
        @gpt_tokenizer = Tokenizers::Tokenizer.from_file(gpt_tokenizer_path.to_s)
      end
    end
  end
end

# Pre-load models on boot in production/development if desired
# Rails.application.config.after_initialize do
#   ModelLoader.instance
# end
