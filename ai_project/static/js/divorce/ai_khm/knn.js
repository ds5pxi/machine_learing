// 최고의 파라미터 테스트
function knnParamsTest() {
    const knnMinVal = document.querySelector("input[id='knnMinVal']").value;
    const knnMaxVal = document.querySelector("input[id='knnMaxVal']").value;
    const knnAddVal = document.querySelector("input[id='knnAddVal']").value;
    const knnTestForm = document.querySelector("form[name='knnTestForm']");

    if ((knnMinVal === undefined) || (knnMinVal === null) || (knnMinVal.trim() === "")) {
        alert("파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((knnMaxVal === undefined) || (knnMaxVal === null) || (knnMaxVal.trim() === "")) {
        alert("파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((knnAddVal === undefined) || (knnAddVal === null) || (knnAddVal.trim() === "")) {
        alert("파라미터 증가값을 입력해 주세요.");

        return;
    }

    knnTestForm.submit();
}

// 최적의 파라미터 입력 후 결과 값
function knnOptParamsTest() {
    const knnNeighbors = document.querySelector("input[id='knnNeighbors']").value;
    const knnOptPrmsForm = document.querySelector("form[name='knnOptPrmsForm']");

    if ((knnNeighbors === undefined) || (knnNeighbors === null) || (knnNeighbors.trim() === "")) {
        alert("n_neighbors 값을 입력해 주세요.");

        return;
    }

    knnOptPrmsForm.submit();
}