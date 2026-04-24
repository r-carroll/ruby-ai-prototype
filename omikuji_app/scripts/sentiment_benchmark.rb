# omikuji_app/scripts/sentiment_benchmark.rb
require_relative '../config/environment'

test_cases = [
  { text: "今日は本当に最高の日です！", expected: :positive, desc: "Enthusiastic Positive" },
  { text: "この映画はとても面白かったです。", expected: :positive, desc: "Standard Positive" },
  { text: "美味しいご飯を食べて幸せです。", expected: :positive, desc: "Positive Food" },
  { text: "最悪な一日でした。二度と行きたくない。", expected: :negative, desc: "Enthusiastic Negative" },
  { text: "料理が冷めていて、美味しくなかった。", expected: :negative, desc: "Standard Negative" },
  { text: "対応が非常に悪く、不快な思いをしました。", expected: :negative, desc: "Service Negative" },
  { text: "普通の味でした。特に言うことはありません。", expected: :negative, desc: "Neutral-ish (Often Negative in reviews)" },
  { text: "おはようございます。", expected: :positive, desc: "Simple Greeting" }
]

puts "BERT Sentiment Benchmark"
puts "========================"

passed = 0
test_cases.each do |tc|
  result = SentimentAnalysisService.new(tc[:text]).predict
  status = result[:label] == tc[:expected] ? "✅" : "❌"
  passed += 1 if result[:label] == tc[:expected]
  
  puts "#{status} [#{tc[:desc]}]"
  puts "   Text: #{tc[:text]}"
  puts "   Detected: #{result[:label].to_s.upcase} (Conf: #{(result[:score] * 100).round(1)}%)"
  puts "   Probabilities (Neg/Pos): #{result[:all_scores].map{|s| s.round(4)}.inspect}"
  puts "-" * 40
end

puts "Summary: #{passed}/#{test_cases.size} tests passed."
