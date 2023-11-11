window.addEventListener('DOMContentLoaded', (event) => {
    var header = document.querySelector('.md-header');
    var tabs = document.querySelector('.md-tabs');

    if (window.scrollY < firstSection.offsetHeight) {
        header.classList.add('md-header_opaque');
        tabs.classList.add('md-tabs_opaque');
    }
});

window.addEventListener('scroll', function () {
    var header = document.querySelector('.md-header');
    var tabs = document.querySelector('.md-tabs');
    var firstSection = document.getElementById('firstSection');

    if (window.scrollY < firstSection.offsetHeight) {
        header.classList.add('md-header_opaque');
        tabs.classList.add('md-tabs_opaque');
    } else {
        header.classList.remove('md-header_opaque');
        tabs.classList.remove('md-tabs_opaque');
    }
});

window.addEventListener('scroll', function () {
    var items = document.querySelectorAll('.landing_second__item');
    var windowHeight = window.innerHeight;

    items.forEach(function (item, index) {
        var itemPosition = item.getBoundingClientRect();

        if (itemPosition.top < windowHeight && itemPosition.bottom >= 0) {
            if (index % 2 === 1) { // Check if the item is even
                item.style.animation = 'flyInRight 2s ease-in-out 1';
            } else { // The item is odd
                item.style.animation = 'flyInLeft 2s ease-in-out 1';
            }
            item.style.animationFillMode = 'forwards';
        }
    });
});
