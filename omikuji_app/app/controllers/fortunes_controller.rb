# omikuji_app/app/controllers/fortunes_controller.rb

class FortunesController < ApplicationController
  def index
    @fortunes = Fortune.order(created_at: :desc).limit(10)
    @fortune = Fortune.new
  end

  def create
    @fortune = Fortune.new(fortune_params)
    @fortune.status = "pending"

    if @fortune.input_text.present?
      if @fortune.save
        FortuneGenerationJob.perform_later(@fortune.id)
        
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

  def status
    # In the new async architecture, we check if there's an active background worker
    # instead of checking if the model is loaded in the web process.
    worker_active = begin
      SolidQueue::Process.active.where(kind: "Worker").any?
    rescue
      false
    end
    
    render json: { loaded: worker_active }
  end

  private

  def fortune_params
    params.require(:fortune).permit(:input_text)
  end
end
