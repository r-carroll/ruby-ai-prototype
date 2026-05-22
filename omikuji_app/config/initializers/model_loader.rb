# config/initializers/model_loader.rb

require 'onnxruntime'
require 'tokenizers'
require 'natto'

class ModelLoader
  include Singleton

  attr_reader :mecab, :markov_service

  def initialize
    @models_path = Rails.root.join('vendor', 'models')
    @mecab = Natto::MeCab.new
    @bert_initialized = false
    @init_lock = Mutex.new

    # Initialize Custom Markov Service immediately (it's fast)
    corpus_path = @models_path.join('fortunes_corpus.txt')
    if File.exist?(corpus_path)
      corpus_text = File.read(corpus_path)
      segmented_corpus = corpus_text.split("\n").map do |line|
        @mecab.parse(line).split("\n").map { |l| l.split("\t").first }.join(" ")
      end.join("\n")
      @markov_service = MarkovService.new(segmented_corpus)
    end
  end

  def bert_session
    ensure_bert_initialized
    @bert_session
  end

  def bert_tokenizer
    ensure_bert_initialized
    @bert_tokenizer
  end

  def bert_loaded?
    @bert_initialized
  end

  private

  def ensure_bert_initialized
    return if @bert_initialized

    @init_lock.synchronize do
      return if @bert_initialized

      start_time = Time.now
      Rails.logger.info "ModelLoader: BERT initialization starting..."
      
      bert_path = @models_path.join('model.onnx')
      tokenizer_json_path = @models_path.join('tokenizer.json')

      if File.exist?(bert_path) && File.exist?(tokenizer_json_path)
        begin
          Rails.logger.info "ModelLoader: Loading ONNX session from #{bert_path}..."
          s_time = Time.now
          @bert_session = OnnxRuntime::Model.new(bert_path.to_s)
          Rails.logger.info "ModelLoader: ONNX session loaded in #{(Time.now - s_time).round(2)}s"

          Rails.logger.info "ModelLoader: Loading Tokenizer from #{tokenizer_json_path}..."
          s_time = Time.now
          # Loading from JSON is handled in native Rust and is sub-second
          @bert_tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_json_path.to_s)
          
          # BERT Japanese v3 needs these specific settings after loading
          @bert_tokenizer.enable_padding(length: 128)
          @bert_tokenizer.enable_truncation(128)
          
          Rails.logger.info "ModelLoader: Tokenizer loaded in #{(Time.now - s_time).round(2)}s"

          @bert_initialized = true
          Rails.logger.info "ModelLoader: BERT initialization complete in #{(Time.now - start_time).round(2)}s"
        rescue => e
          Rails.logger.error "ModelLoader: BERT initialization failed: #{e.message}\n#{e.backtrace.join("\n")}"
        end
      else
        Rails.logger.error "ModelLoader: BERT files missing at #{@models_path}. Expected model.onnx and tokenizer.json"
      end
    end
  end
end

# Pre-loading is now handled explicitly by the background worker in bin/jobs
# to prevent blocking the web server during cold starts.
