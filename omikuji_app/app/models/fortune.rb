class Fortune < ApplicationRecord
  validates :input_text, presence: true

  after_update_commit -> { 
    # Update the main show partial
    broadcast_replace_to self, 
      target: "fortune_#{id}", 
      partial: "fortunes/show", 
      locals: { fortune: self } 

    # Update the list item partial
    broadcast_replace_to self, "list",
      target: "fortune_list_#{id}",
      partial: "fortunes/fortune_list_item",
      locals: { fortune: self }
  }
end
