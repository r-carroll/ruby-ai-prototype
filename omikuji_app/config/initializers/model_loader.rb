# config/initializers/model_loader.rb

require 'onnxruntime'
require 'tokenizers'
require 'natto'

class ModelLoader
  include Singleton

  attr_reader :bert_session, :bert_tokenizer, :mecab, :markov_service

  def initialize
    @models_path = Rails.root.join('vendor', 'models')
    @mecab = Natto::MeCab.new
    
    # Load BERT Sentiment Model
    bert_path = @models_path.join('model.onnx')
    vocab_path = @models_path.join('vocab.txt')
    
    if File.exist?(bert_path) && File.exist?(vocab_path)
      @bert_session = OnnxRuntime::Model.new(bert_path.to_s)
      vocab = File.readlines(vocab_path).each_with_index.to_h { |line, i| [line.strip, i] }
      @bert_tokenizer = Tokenizers::Tokenizer.new(Tokenizers::Models::WordPiece.new(vocab: vocab))
      @bert_tokenizer.normalizer = Tokenizers::Normalizers::BertNormalizer.new
      @bert_tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::BertPreTokenizer.new

      # Add Post-Processor for [CLS] and [SEP]
      @bert_tokenizer.post_processor = Tokenizers::Processors::TemplateProcessing.new(
        single: "[CLS] $A [SEP]",
        pair: "[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens: [
          ["[CLS]", 2],
          ["[SEP]", 3]
        ]
      )
    end

    # Initialize Custom Markov Service
    corpus_path = @models_path.join('fortunes_corpus.txt')
    if File.exist?(corpus_path)
      corpus_text = File.read(corpus_path)
      
      # BERT Japanese models expect text to be pre-segmented by MeCab
      # We do the same for Markov Chain training
      segmented_corpus = corpus_text.split("\n").map do |line|
        @mecab.parse(line).split("\n").map { |l| l.split("\t").first }.join(" ")
      end.join("\n")
      
      @markov_service = MarkovService.new(segmented_corpus)
    end
  end
end

# Pre-load models on boot in a background thread to prevent blocking Rails boot.
# This allows the server to start and pass health checks immediately.
Rails.application.config.after_initialize do
  Thread.new do
    begin
      ModelLoader.instance
      Rails.logger.info "ModelLoader: Background initialization complete."
    rescue => e
      Rails.logger.error "ModelLoader: Background initialization failed: #{e.message}"
    end
  end
end
