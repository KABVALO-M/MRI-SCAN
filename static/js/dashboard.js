// Dropdown toggle functionality
const dropdownToggle = document.getElementById('dropdownToggle');
const dropdownMenu = document.getElementById('dropdownMenu');

dropdownToggle.addEventListener('click', () => {
    dropdownMenu.classList.toggle('hidden');
});