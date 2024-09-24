document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-upload');
    const fileUploadText = document.getElementById('file-upload-text');

    fileInput.addEventListener('change', function() {
        const fileName = fileInput.files[0] ? fileInput.files[0].name : 'Click to upload or drag and drop';
        fileUploadText.innerHTML = `<span class="font-semibold">${fileName}</span>`;
    });

    // Optional: Drag and drop functionality to show file name
    const label = fileInput.closest('label');

    label.addEventListener('dragover', function(event) {
        event.preventDefault();
        label.classList.add('bg-gray-200');
    });

    label.addEventListener('dragleave', function() {
        label.classList.remove('bg-gray-200');
    });

    label.addEventListener('drop', function(event) {
        event.preventDefault();
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files; // Assign the dropped files to the input
            fileUploadText.innerHTML = `<span class="font-semibold">${files[0].name}</span>`; // Update text
        }
        label.classList.remove('bg-gray-200');
    });
});