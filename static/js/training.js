const form = document.querySelector('form');
const modal = document.getElementById('loading-modal');

form.addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent form from submitting the traditional way

    // Show loading modal
    modal.classList.remove('hidden');

    const formData = new FormData(form);

    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest', // This tells the view it's an AJAX request
        },
    })
    .then(response => {
        if (response.ok) {
            return response.json(); // Expecting a JSON response from the server
        } else {
            throw new Error('Something went wrong during training.');
        }
    })
    .then(data => {
        if (data.success) {
            // Redirect to another URL, passing the history data
            const historyJson = JSON.stringify(data.history);
            window.location.href = `/training-results/?history=${encodeURIComponent(historyJson)}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        modal.classList.add('hidden'); // Hide modal in case of error
        alert('An error occurred while starting the training. Please try again.');
    });
});


// ==========================================================================================================================================================================
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
