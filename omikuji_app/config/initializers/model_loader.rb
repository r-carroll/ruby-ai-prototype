# config/initializers/model_loader.rb

require 'onnxruntime'
require 'tokenizers'

class ModelLoader
  include Singleton

  attr_reader :bert_session, :tokenizer

  def initialize
    @models_path = Rails.root.join('vendor', 'models')
    
    # Load BERT Sentiment Model
    bert_path = @models_path.join('model.onnx')
    if File.exist?(bert_path)
      Rails.logger.info "Loading BERT model from #{bert_path}..."
      @bert_session = OnnxRuntime::Model.new(bert_path.to_s)
    else
      Rails.logger.warn "BERT model not found at #{bert_path}. Inference will be disabled."
    end

    # Load Tokenizer
    vocab_path = @models_path.join('vocab.txt')
    if File.exist?(vocab_path)
      # Note: We'll refine this once we verify if we use vocab.txt or a tokenizer.json
      # For now, we assume WordPiece tokenizer setup
      @tokenizer = Tokenizers::BertTokenizer.from_file(vocab_path.to_s)
    else
      Rails.logger.warn "Tokenizer vocab not found at #{vocab_path}."
    end
  end
end

# Pre-load models on boot in production/development if desired
# Rails.application.config.after_initialize do
#   ModelLoader.instance
# end
