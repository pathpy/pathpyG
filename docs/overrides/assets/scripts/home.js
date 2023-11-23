window.addEventListener('scroll', function () {
    var header = document.querySelector('.md-header');
    var firstSection = document.getElementById('firstSection');

    if (window.scrollY > firstSection.offsetHeight) {
        header.classList.add('md-bg-color');
    } else {
        header.classList.remove('md-bg-color');
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

var hasScrolledPast = false;

window.addEventListener('scroll', function() {
    var heroContent = document.getElementById('hero-content');
    var imageElement = document.querySelector('.parallax__image');
    var imageHeight = imageElement ? imageElement.offsetHeight : 0;
    var scrollPosition = window.scrollY;

    if (scrollPosition > imageHeight / 2) {
        heroContent.style.animation = "fadeOutUp 1s forwards";
        hasScrolledPast = true;
    } else if (hasScrolledPast) {
        heroContent.style.animation = "fadeInDown 1s forwards";
    }
});

/* Affiliations */

$(document).ready(function(){
    $("#affiliation-slider").owlCarousel({
        items: 3,
        itemsDesktop: [1220, 3],
        itemsDesktopSmall: [960, 2],
        itemsTablet: [768, 1],
        loop:true,
        nav: true,
        autoplay:true,
        autoplayHoverPause:true,
        dotsEach: 1,
    });
});
