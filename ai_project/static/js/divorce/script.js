document.addEventListener('DOMContentLoaded', function () {
    // 모든 질문을 비활성화하고 첫 번째 질문만 활성화
    const questions = document.querySelectorAll('.question');
    questions.forEach((question, index) => {
        if (index !== 0) {
            question.style.pointerEvents = 'none';
            question.style.opacity = '0.5';
        }
    });

    // 라디오 버튼 클릭 시 이벤트 처리
    document.querySelectorAll('.custom-checkbox-group input[type="radio"]').forEach(function(radio) {
        radio.addEventListener('click', function() {
            const currentQuestion = this.closest('.question');
            const nextQuestion = currentQuestion.nextElementSibling;
            if (nextQuestion && nextQuestion.classList.contains('question')) {
                // 다음 질문을 활성화
                nextQuestion.style.pointerEvents = 'auto';
                nextQuestion.style.opacity = '1';

                // 다음 질문으로 스크롤 이동
                nextQuestion.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    });
});
