class CreateFortunes < ActiveRecord::Migration[8.1]
  def change
    create_table :fortunes do |t|
      t.text :input_text
      t.string :sentiment_label
      t.text :fortune_text
      t.string :rank
      t.float :score

      t.timestamps
    end
  end
end
