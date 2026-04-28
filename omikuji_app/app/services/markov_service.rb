# omikuji_app/app/services/markov_service.rb

class MarkovService
  def initialize(segmented_corpus)
    @chain = Hash.new { |h, k| h[k] = [] }
    @starts = []
    train(segmented_corpus)
  end

  # Trains the model using space-separated Japanese tokens
  def train(corpus)
    corpus.split("\n").each do |line|
      tokens = line.strip.split(" ")
      next if tokens.size < 2

      # Store the first two words as a potential starting state
      @starts << tokens[0..1]

      # Build the chain using a 2-word state (Bigram)
      tokens.each_cons(3) do |w1, w2, w3|
        @chain[[w1, w2]] << w3
      end
    end
  end

  # Generates a single "natural" sounding sentence
  def generate_sentence(max_tokens = 30)
    return "" if @starts.empty?

    state = @starts.sample
    result = state.dup

    max_tokens.times do
      next_word = @chain[state].sample
      break if next_word.nil?

      result << next_word
      state = [state[1], next_word]
      
      # Stop if we hit a sentence-ending punctuation
      break if next_word.match?(/[。！？]/)
    end

    result.join("")
  end
end
