document.addEventListener("DOMContentLoaded", function () {
    // Get the file input and upload label elements
    const fileInput = document.getElementById('file-upload');
    const fileUploadText = document.getElementById('file-upload-text');

    // File types allowed
    const allowedFileTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    const maxFileSize = 5 * 1024 * 1024;  // 5MB

    // Update the label text when a file is selected
    fileInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        
        if (file) {
            // Check file type
            if (!allowedFileTypes.includes(file.type)) {
                alert('Invalid file type! Only JPG, PNG, and BMP are supported.');
                fileInput.value = ''; // Reset the input
                fileUploadText.innerText = 'Click to upload or drag and drop';
                return;
            }

            // Check file size
            if (file.size > maxFileSize) {
                alert('File is too large! Max size is 5MB.');
                fileInput.value = ''; // Reset the input
                fileUploadText.innerText = 'Click to upload or drag and drop';
                return;
            }

            // Display the file name in the label
            fileUploadText.innerText = file.name;
        } else {
            fileUploadText.innerText = 'Click to upload or drag and drop';
        }
    });

    // Prevent default behavior for drag-and-drop
    const dropArea = document.querySelector('.w-full.h-32.border-dashed');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight the drop area when dragging files over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('border-blue-400');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('border-blue-400');
        }, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', function (event) {
        let files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files; // Assign dropped files to the file input
            const file = files[0];

            // Check file type
            if (!allowedFileTypes.includes(file.type)) {
                alert('Invalid file type! Only JPG, PNG, and BMP are supported.');
                fileInput.value = ''; // Reset the input
                fileUploadText.innerText = 'Click to upload or drag and drop';
                return;
            }

            // Check file size
            if (file.size > maxFileSize) {
                alert('File is too large! Max size is 5MB.');
                fileInput.value = ''; // Reset the input
                fileUploadText.innerText = 'Click to upload or drag and drop';
                return;
            }

            // Display the file name in the label
            fileUploadText.innerText = file.name;
        }
    });
});
