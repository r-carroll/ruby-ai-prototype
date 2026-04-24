import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["form", "button", "box", "loading"]

  shake(event) {
    // 1. Prevent multiple clicks
    this.buttonTarget.disabled = true
    
    // 2. Start shaking animation
    this.boxTarget.classList.add("animate-shake")
    
    // 3. Show loading message
    this.loadingTarget.classList.remove("hidden")
  }

  // Called via Turbo Stream or event when the result is returned
  reset() {
    this.boxTarget.classList.remove("animate-shake")
    this.buttonTarget.disabled = false
    this.loadingTarget.classList.add("hidden")
    
    // Clear the text area for the next draw
    const textArea = this.element.querySelector("textarea")
    if (textArea) textArea.value = ""
  }
}
