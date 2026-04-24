# omikuji_app/app/controllers/fortunes_controller.rb

class FortunesController < ApplicationController
  def index
    @fortunes = Fortune.order(created_at: :desc).limit(10)
    @fortune = Fortune.new
  end

  def create
    @fortune = Fortune.new(fortune_params)

    if @fortune.input_text.present?
      # 1. Run Sentiment Analysis (The "Ear")
      sentiment_result = SentimentAnalysisService.new(@fortune.input_text).predict
      @fortune.sentiment_label = sentiment_result[:label].to_s
      @fortune.score = sentiment_result[:score]

      # 2. Run Fortune Generation (The "Voice")
      generation_result = FortuneGeneratorService.new(sentiment_result[:label]).generate
      @fortune.fortune_text = generation_result[:fortune]
      @fortune.rank = generation_result[:rank]

      if @fortune.save
        respond_to do |format|
          format.html { redirect_to fortune_path(@fortune) }
          format.turbo_stream
        end
      else
        render :index, status: :unprocessable_entity
      end
    else
      redirect_to root_path, alert: "Please enter some text."
    end
  end

  def show
    @fortune = Fortune.find(params[:id])
  end

  private

  def fortune_params
    params.require(:fortune).permit(:input_text)
  end
end
