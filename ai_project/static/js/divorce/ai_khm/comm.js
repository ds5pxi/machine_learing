// 머신러닝, 딥러닝 이혼 사이트로 돌아가기(공통 사용하므로 여기로 빼냄)
function returnLearning() {
    location.href = "/ai_khm/divorce/ai_khm/learning";
}

// 숫자 값 외에는 입력 안 받는 함수
function onlyInputNumber(thisObj) {
    thisObj.value = thisObj.value.replace(/[^0-9]/g, '');
}