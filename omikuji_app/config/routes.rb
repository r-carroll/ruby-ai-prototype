Rails.application.routes.draw do
  root "fortunes#index"
  resources :fortunes, only: [:index, :create, :show]
  
  get "up" => "rails/health#show", as: :rails_health_check
end
