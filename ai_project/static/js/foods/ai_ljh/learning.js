function validResult() {
    const emotionList = document.querySelectorAll("input[type='radio'][id^='k-emotion']");
    const seasonList = document.querySelectorAll("input[type='radio'][id^='k-season']");
    const weatherList = document.querySelectorAll("input[type='radio'][id^='k-weather']");
    const peopleList = document.querySelectorAll("input[type='radio'][id^='k-people']");
    const priceList = document.querySelectorAll("input[type='radio'][id^='k-price']");
    const timeList = document.querySelectorAll("input[type='radio'][id^='k-time']");
    const sexList = document.querySelectorAll("input[type='radio'][id^='k-sex']");
    const currForm = document.querySelector("form")

    let answerArr = [];

    for (const chkFlag of emotionList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of seasonList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of weatherList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of peopleList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of priceList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of timeList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    for (const chkFlag of sexList) {
        if (chkFlag.checked) {
            answerArr.push(chkFlag.value);
        }
    }

    if (answerArr.length != 7) {
        alert("메뉴를 체크해 주세요.");

        return;
    }

    if (window.confirm("추천 음식 결과를 보시겠습니까?")) {
        currForm.submit();
    }
}

// static/js/foods/ai_ljh/learning.js

function showEmoji(emoji, targetElement) {
    const emojiSpan = document.createElement('span');
    emojiSpan.textContent = emoji;
    emojiSpan.classList.add('emoji-pop');

    const rect = targetElement.getBoundingClientRect();
    emojiSpan.style.left = rect.left + window.pageXOffset + 'px';
    emojiSpan.style.top = rect.top + window.pageYOffset + 'px';

    document.body.appendChild(emojiSpan);

    setTimeout(() => {
        emojiSpan.remove();
    }, 800);
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.form-check-input').forEach(input => {
        input.addEventListener('change', (event) => {
            const emojiMap = {
                'k-emotion1': '😚',
                'k-emotion2': '😢',
                'k-emotion3': '🤬',
                'k-emotion4': '😩',
                'k-season1': '🌷',
                'k-season2': '☀️',
                'k-season3': '🍁',
                'k-season4': '❄️',
                'k-weather1': '☀️',
                'k-weather2': '🌥️',
                'k-weather3': '🌧️',
                'k-weather4': '❄️',
                'k-people1': '😎',
                'k-people2': '🧑‍🤝‍🧑',
                'k-people3': '👨‍👩‍👧',
                'k-people4': '👨‍👩‍👧‍👦',
                'k-price1': '🪙',
                'k-price2': '💵',
                'k-price3': '💸',
                'k-time1': '🌞',
                'k-time2': '🌜',
                'k-sex1': '👨',
                'k-sex2': '👩',
                'k-sex3': '👫'
            };
            showEmoji(emojiMap[event.target.id], event.target);
        });
    });
});