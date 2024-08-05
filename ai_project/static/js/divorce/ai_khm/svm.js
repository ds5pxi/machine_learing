// SVM 모델 결과 보기
function svmResult() {
    location.href = "/ai_khm/divorce/ai_khm/svm";
}

// SVM 교차 검증 보기
function svmCrossValidation() {
    location.href = "/ai_khm/divorce/ai_khm/svm_cv";
}

// SVM 최고의 파라미터 테스트
function svmParamsTest() {
    const svmMinVal = document.querySelector("input[id='svmMinVal']").value;
    const svmMaxVal = document.querySelector("input[id='svmMaxVal']").value;
    const svmAddVal = document.querySelector("input[id='svmAddVal']").value;
    const svmTestForm = document.querySelector("form[name='svmTestForm']");

    if ((svmMinVal === undefined) || (svmMinVal === null) || (svmMinVal.trim() === "")) {
        alert("파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((svmMaxVal === undefined) || (svmMaxVal === null) || (svmMaxVal.trim() === "")) {
        alert("파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((svmAddVal === undefined) || (svmAddVal === null) || (svmAddVal.trim() === "")) {
        alert("파라미터 증가값을 입력해 주세요.");

        return;
    }

    svmTestForm.submit();
}

// 최적의 파라미터 입력 후 결과 값
function svmOptParamsTest() {
    const svmC = document.querySelector("input[id='svmC']").value;
    const svmOptPrmsForm = document.querySelector("form[name='svmOptPrmsForm']");

    if ((svmC === undefined) || (svmC === null) || (svmC.trim() === "")) {
        alert("C 값을 입력해 주세요.");

        return;
    }

    svmOptPrmsForm.submit();
}