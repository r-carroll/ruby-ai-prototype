Rails.application.routes.draw do
  get "pages/architecture"
  root "fortunes#index"
  resources :fortunes, only: [:index, :create, :show]
  
  get "architecture", to: "pages#architecture"
  get "up" => "rails/health#show", as: :rails_health_check
end
