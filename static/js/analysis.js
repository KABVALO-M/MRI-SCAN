
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const output = document.getElementById('image_preview');
        output.src = reader.result;
    }
    reader.readAsDataURL(event.target.files[0]);
}


// Function to update the fields based on the patient code
function updatePatientFields() {
    // Get the entered patient code from the input field
    const patientCode = document.getElementById('patient_code').value;

    // Search for a matching patient in the patients array
    const matchingPatient = patients.find(patient => patient.fields.patient_code === patientCode);

    if (matchingPatient) {
        // If a matching patient is found, update the form fields
        document.getElementById('first_name').value = matchingPatient.fields.first_name;
        document.getElementById('last_name').value = matchingPatient.fields.last_name;
        document.getElementById('age').value = matchingPatient.fields.age;
        document.getElementById('gender').value = matchingPatient.fields.gender;
    } else {
        // If no matching patient is found, clear the fields
        document.getElementById('first_name').value = '';
        document.getElementById('last_name').value = '';
        document.getElementById('age').value = '';
        document.getElementById('gender').value = '';
    }
}

// Attach the function to the onchange or oninput event of the patient code input
document.getElementById('patient_code').oninput = updatePatientFields;



function checkFields() {
    const patientCode = document.getElementById('patient_code').value.trim();
    const firstName = document.getElementById('first_name').value.trim();
    const lastName = document.getElementById('last_name').value.trim();
    const age = document.getElementById('age').value.trim();
    const gender = document.getElementById('gender').value.trim();
    const imageInput = document.getElementById('image').value.trim();

    // Get the Analyse button
    const analyseButton = document.getElementById('analyseButton');

    // Check if all fields are filled
    if (patientCode && firstName && lastName && age && gender && imageInput) {
        // Enable the button if all fields are filled
        analyseButton.disabled = false;
    } else {
        // Disable the button if any field is empty
        analyseButton.disabled = true;
    }
}


document.addEventListener("DOMContentLoaded", function () {
    // Get the file input and upload label elements
    const fileInput = document.getElementById('image'); // Updated to match the new input ID
    const fileUploadText = document.getElementById('image-upload-text'); // Updated to match the new ID

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
    const dropArea = document.querySelector('.border-dashed'); // Updated to use the dashed border class
    
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
