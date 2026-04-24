require 'rails_helper'

RSpec.describe "Fortunes", type: :request do
  describe "GET /index" do
    it "returns http success" do
      get "/fortunes/index"
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET /create" do
    it "returns http success" do
      get "/fortunes/create"
      expect(response).to have_http_status(:success)
    end
  end

  describe "GET /show" do
    it "returns http success" do
      get "/fortunes/show"
      expect(response).to have_http_status(:success)
    end
  end

end
