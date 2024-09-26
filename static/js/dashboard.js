// Dropdown toggle functionality
const dropdownToggle = document.getElementById('dropdownToggle');
const dropdownMenu = document.getElementById('dropdownMenu');

dropdownToggle.addEventListener('click', () => {
    dropdownMenu.classList.toggle('hidden');
});

// Function to automatically close message divs after a specified duration
function autoCloseMessages(duration = 5000) {
    const messages = document.querySelectorAll('.message');
    console.log("Found messages:", messages.length); 
    messages.forEach(function(message) {
        setTimeout(function() {
            console.log("Closing Message:", message); 
            message.style.display = 'none';
        }, duration);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    autoCloseMessages(); 
});
