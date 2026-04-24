import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static values = { url: String }
  static targets = ["label"]

  copy(event) {
    event.preventDefault()
    
    navigator.clipboard.writeText(this.urlValue).then(() => {
      const originalText = this.labelTarget.textContent
      this.labelTarget.innerText = "Link Copied! ✓"
      this.labelTarget.classList.add("text-green-600")
      
      setTimeout(() => {
        this.labelTarget.innerText = originalText
        this.labelTarget.classList.remove("text-green-600")
      }, 2000)
    })
  }
}
