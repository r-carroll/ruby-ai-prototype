# omikuji_app/spec/services/fortune_generator_service_spec.rb

require 'rails_helper'

RSpec.describe FortuneGeneratorService do
  describe "#generate" do
    it "generates a fortune for positive sentiment" do
      service = FortuneGeneratorService.new(:positive)
      result = service.generate

      expect(result).to have_key(:rank)
      expect(result).to have_key(:fortune)
      expect(result[:fortune]).to start_with("「")
      puts "Generated Positive Fortune: #{result[:rank]} - #{result[:fortune]}"
    end

    it "generates a fortune for negative sentiment" do
      service = FortuneGeneratorService.new(:negative)
      result = service.generate

      expect(result[:fortune]).to be_a(String)
      puts "Generated Negative Fortune: #{result[:rank]} - #{result[:fortune]}"
    end
  end
end
