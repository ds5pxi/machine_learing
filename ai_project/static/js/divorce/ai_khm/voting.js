// 보팅 최고의 파라미터 테스트
function votingParamsTest() {
    const votingCMinVal = document.querySelector("input[id='votingCMinVal']").value;
    const votingCMaxVal = document.querySelector("input[id='votingCMaxVal']").value;
    const votingCAddVal = document.querySelector("input[id='votingCAddVal']").value;
    const votingNeighborsMinVal = document.querySelector("input[id='votingNeighborsMinVal']").value;
    const votingNeighborsMaxVal = document.querySelector("input[id='votingNeighborsMaxVal']").value;
    const votingNeighborsAddVal = document.querySelector("input[id='votingNeighborsAddVal']").value;
    const votingMaxDepthMinVal = document.querySelector("input[id='votingMaxDepthMinVal']").value;
    const votingMaxDepthMaxVal = document.querySelector("input[id='votingMaxDepthMaxVal']").value;
    const votingMaxDepthAddVal = document.querySelector("input[id='votingMaxDepthAddVal']").value;
    const votingMinSamplesSplMinVal = document.querySelector("input[id='votingMinSamplesSplMinVal']").value;
    const votingMinSamplesSplMaxVal = document.querySelector("input[id='votingMinSamplesSplMaxVal']").value;
    const votingMinSamplesSplAddVal = document.querySelector("input[id='votingMinSamplesSplAddVal']").value;
    const votingTestForm = document.querySelector("form[name='votingTestForm']");

    if ((votingCMinVal === undefined) || (votingCMinVal === null) || (votingCMinVal.trim() === "")) {
        alert("C 파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((votingCMaxVal === undefined) || (votingCMaxVal === null) || (votingCMaxVal.trim() === "")) {
        alert("C 파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((votingCAddVal === undefined) || (votingCAddVal === null) || (votingCAddVal.trim() === "")) {
        alert("C 파라미터 증가값을 입력해 주세요.");

        return;
    }

    if ((votingNeighborsMinVal === undefined) || (votingNeighborsMinVal === null) || (votingNeighborsMinVal.trim() === "")) {
        alert("n_neighbors 파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((votingNeighborsMaxVal === undefined) || (votingNeighborsMaxVal === null) || (votingNeighborsMaxVal.trim() === "")) {
        alert("n_neighbors 파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((votingNeighborsAddVal === undefined) || (votingNeighborsAddVal === null) || (votingNeighborsAddVal.trim() === "")) {
        alert("n_neighbors 파라미터 증가값을 입력해 주세요.");

        return;
    }

    if ((votingMaxDepthMinVal === undefined) || (votingMaxDepthMinVal === null) || (votingMaxDepthMinVal.trim() === "")) {
        alert("max_depth 파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((votingMaxDepthMaxVal === undefined) || (votingMaxDepthMaxVal === null) || (votingMaxDepthMaxVal.trim() === "")) {
        alert("max_depth 파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((votingMaxDepthAddVal === undefined) || (votingMaxDepthAddVal === null) || (votingMaxDepthAddVal.trim() === "")) {
        alert("max_depth 파라미터 증가값을 입력해 주세요.");

        return;
    }

    if ((votingMinSamplesSplMinVal === undefined) || (votingMinSamplesSplMinVal === null) || (votingMinSamplesSplMinVal.trim() === "")) {
        alert("min_samples_split 파라미터 최소값을 입력해 주세요.");

        return;
    }

    if ((votingMinSamplesSplMaxVal === undefined) || (votingMinSamplesSplMaxVal === null) || (votingMinSamplesSplMaxVal.trim() === "")) {
        alert("min_samples_split 파라미터 최대값을 입력해 주세요.");

        return;
    }

    if ((votingMinSamplesSplAddVal === undefined) || (votingMinSamplesSplAddVal === null) || (votingMinSamplesSplAddVal.trim() === "")) {
        alert("min_samples_split 파라미터 증가값을 입력해 주세요.");

        return;
    }

    votingTestForm.submit();
}