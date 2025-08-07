function previewImage(event) {
    var output = document.getElementById('outputImage');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.style.display = 'block';
}
