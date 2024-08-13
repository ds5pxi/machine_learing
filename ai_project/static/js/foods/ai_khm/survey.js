function clickSurveyLabel(thisEle) {
    const thisNm = thisEle.getAttribute("for");
    const labelList = document.querySelectorAll("label[for^='" + thisNm.substring(0, thisNm.length - 1) + "']");
    const inputList = document.querySelectorAll("input[type='radio'][id^='" + thisNm.substring(0, thisNm.length - 1) + "']");

    for (const label of labelList) {
        label.classList.remove("SelectMenuRadioFilter_active__6BtVl");
    }

    for (const input of inputList) {
        input.removeAttribute("checked");
    }

    thisEle.classList.add("SelectMenuRadioFilter_active__6BtVl");
    document.querySelector("input[type='radio'][id='" + thisNm + "']").setAttribute("checked", "");
}

function validResult() {
    const emotionList = document.querySelectorAll("input[type='radio'][id^='emotion']");
    const seasonList = document.querySelectorAll("input[type='radio'][id^='season']");
    const weatherList = document.querySelectorAll("input[type='radio'][id^='weather']");
    const peopleList = document.querySelectorAll("input[type='radio'][id^='people']");
    const priceList = document.querySelectorAll("input[type='radio'][id^='price']");
    const timeList = document.querySelectorAll("input[type='radio'][id^='time']");
    const sexList = document.querySelectorAll("input[type='radio'][id^='sex']");
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

