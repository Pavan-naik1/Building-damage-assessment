
const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const dropZone = document.getElementById("dropZone");

if (input && preview) {
    input.addEventListener("change", () => {
        const file = input.files && input.files[0];

        if (!file) {
            preview.classList.add("hidden");
            preview.removeAttribute("src");
            return;
        }

        preview.src = URL.createObjectURL(file);
        preview.classList.remove("hidden");
    });
}

if (dropZone) {
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });
}
