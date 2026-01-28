require 'natto'

class JapaneseBertTokenizer
  attr_reader :vocab, :ids_to_tokens

  # Special Tokens
  TOKEN_CLS = "[CLS]"
  TOKEN_SEP = "[SEP]"
  TOKEN_PAD = "[PAD]"
  TOKEN_UNK = "[UNK]"

  def initialize(vocab_path = File.join(__dir__, "../models/vocab.txt"))
    @vocab = load_vocab(vocab_path)
    @ids_to_tokens = @vocab.invert
    
    # Pre-fetch IDs for special tokens
    @cls_id = @vocab[TOKEN_CLS]
    @sep_id = @vocab[TOKEN_SEP]
    @pad_id = @vocab[TOKEN_PAD]
    @unk_id = @vocab[TOKEN_UNK]

    # Initialize MeCab
    # NOTE: This relies on the system mecab dictionary which might differ 
    # from the one used during pre-training (unidic-lite vs ipadic).
    # This may cause slight tokenization mismatches.
    @mecab = Natto::MeCab.new
  end

  def encode(text, max_length: 128)
    # 1. Normalize (Approximation of HF BasicTokenizer)
    text = text.unicode_normalize(:nfkc)
    
    # 2. MeCab Tokenization
    # We just want the surface forms.
    # Note: HF tokenizer might do additional splitting on punctuation/spaces here.
    mecab_tokens = []
    @mecab.parse(text) do |n|
      # Skip beginning/end of sentence markers from MeCab if needed, 
      # though Natto usually iterates nodes.
      # Skip EOS/BOS nodes which have surface=""
      next if n.surface.empty?
      mecab_tokens << n.surface
    end

    # 3. WordPiece Tokenization
    subword_tokens = []
    mecab_tokens.each do |token|
      subwords = wordpiece_tokenize(token)
      subword_tokens.concat(subwords)
    end

    # 4. Truncate (Account for [CLS] and [SEP])
    # Available space for tokens
    max_tokens = max_length - 2
    if subword_tokens.length > max_tokens
      subword_tokens = subword_tokens[0...max_tokens]
    end

    # 5. Build IDs
    input_ids = []
    input_ids << @cls_id
    subword_tokens.each do |token|
      input_ids << (@vocab[token] || @unk_id)
    end
    input_ids << @sep_id

    # 6. Padding & Mask
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = Array.new(input_ids.length, 1)

    # Pad
    padding_length = max_length - input_ids.length
    if padding_length > 0
      input_ids.concat(Array.new(padding_length, @pad_id))
      attention_mask.concat(Array.new(padding_length, 0))
    end

    {
      input_ids: input_ids,
      attention_mask: attention_mask
    }
  end

  private

  def load_vocab(path)
    vocab = {}
    File.foreach(path).with_index do |line, index|
      token = line.chomp
      vocab[token] = index
    end
    vocab
  end

  # Greedy WordPiece Algorithm
  def wordpiece_tokenize(token)
    output_tokens = []
    start = 0
    token_len = token.length

    while start < token_len
      err = true
      end_idx = token_len

      while end_idx > start
        sub_token = token[start...end_idx]
        
        # Add '##' prefix for non-initial subwords
        current_sub_token = (start > 0) ? "##{sub_token}" : sub_token

        if @vocab.key?(current_sub_token)
          output_tokens << current_sub_token
          start = end_idx
          err = false
          break
        end

        end_idx -= 1
      end

      if err
        # If no match found for any prefix, the whole token is UNK.
        # HF behavior: maps the whole original token to [UNK] and stops processing it.
        return [TOKEN_UNK]
      end
    end

    output_tokens
  end
end
