import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["form", "button", "box", "loading"]
  static values = { statusUrl: String }

  async shake(event) {
    // Normal submission flow - immediate submission is fine now
    // as the background worker handles the 40s model loading.
    this.buttonTarget.disabled = true
    this.boxTarget.classList.add("animate-shake")
    this.loadingTarget.classList.remove("hidden")
  }

  disconnect() {
    if (this.pollInterval) clearInterval(this.pollInterval)
  }

  reset() {
    this.boxTarget.classList.remove("animate-shake")
    this.boxTarget.classList.remove("animate-pulse")
    this.buttonTarget.disabled = false
    this.loadingTarget.classList.add("hidden")
    
    const textArea = this.element.querySelector("textarea")
    if (textArea) textArea.value = ""
  }
}
