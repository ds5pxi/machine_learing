// 숫자 값 외에는 입력 안 받는 함수
function onlyInputNumber(thisObj) {
    thisObj.value = thisObj.value.replace(/[^0-9]/g, '');
}

// 숫자와 마침표 외에는 입력 안 받는 함수
function onlyInputNumberAndDot(thisObj) {
    thisObj.value = thisObj.value.replace(/[^0-9.]/g, '');
}