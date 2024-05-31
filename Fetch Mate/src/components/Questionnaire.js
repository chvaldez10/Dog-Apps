import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import ProgressBar from "react-progressbar";
import './Questionnaire.css';

const questionsPerPage = 3; // Number of questions per page

function Questionnaire({ questions }) {
    const categories = ["appearance", "personality", "lifestyle"]; // Define categories
    const totalCategoryPages = categories.map(category => {
        const categoryQuestions = questions.filter(question => question.category === category);
        return Math.ceil(categoryQuestions.length / questionsPerPage);
    });

    const [currentPage, setCurrentPage] = useState(1);
    const [categoryIndex, setCategoryIndex] = useState(0); // Track the current category index
    const [answers, setAnswers] = useState(categories.map(category => Array(questions.filter(question => question.category === category).length).fill('')));
    const navigate = useNavigate();

    const totalPages = totalCategoryPages[categoryIndex];

    const nextPage = () => {
        if (currentPage < totalPages) {
            setCurrentPage(currentPage + 1);
        } else {
            if (categoryIndex < categories.length - 1) {
                setCurrentPage(1);
                setCategoryIndex(categoryIndex + 1);
            } else {
                navigate("/register");
            }
        }
    };

    const prevPage = () => {
        if (currentPage > 1) {
            setCurrentPage(currentPage - 1);
        } else {
            if (categoryIndex > 0) {
                setCategoryIndex(categoryIndex - 1);
                setCurrentPage(totalCategoryPages[categoryIndex - 1]);
            }
        }
    };

    const handleAnswerChange = (questionIndex, value, type) => {
        setAnswers(prevAnswers => {
            const newAnswers = [...prevAnswers];
            newAnswers[categoryIndex][questionIndex] = value;
            return newAnswers;
        });
    };    

    const renderQuestions = () => {
        const startIndex = (currentPage - 1) * questionsPerPage;
        const endIndex = currentPage * questionsPerPage;

        const currentCategory = categories[categoryIndex];
        const filteredQuestions = questions.filter(question => question.category === currentCategory);

        const visibleQuestions = filteredQuestions.slice(startIndex, endIndex);

        return visibleQuestions.map((question, index) => {
            return renderQuestion(question, startIndex + index);
        });
    };

    const renderQuestion = (question, questionIndex) => {
        const { question: questionText, type, choices } = question;
        
        const isSelected = (choice) => {
            return answers[categoryIndex][questionIndex].includes(choice);
        };
    
        switch (type) {
            case "single-select":
                return (
                    <div className="question" key={questionIndex}>
                        <p>{questionText}</p>
                        <div className="options">
                            {choices.map((choice, choiceIndex) => (
                                <div
                                    key={choiceIndex}
                                    className={`option ${answers[categoryIndex][questionIndex] === choice ? 'selected' : ''}`}
                                    onClick={() => handleAnswerChange(questionIndex, choice, type)}
                                >
                                    {choice}
                                </div>
                            ))}
                        </div>
                    </div>
                );
            case "multiple-select":
                return (
                    <div className="question" key={questionIndex}>
                        <p>{questionText}</p>
                        <div className="options">
                            {choices.map((choice, choiceIndex) => (
                                <div
                                    key={choiceIndex}
                                    className={`option ${isSelected(choice) ? 'selected' : ''}`}
                                    onClick={() => {
                                        const selectedOptions = [...answers[categoryIndex][questionIndex]];
                                        const selectedIndex = selectedOptions.indexOf(choice);
    
                                        if (selectedIndex === -1) {
                                            // If choice is not selected, add it to the array
                                            selectedOptions.push(choice);
                                        } else {
                                            // If choice is already selected, remove it from the array
                                            selectedOptions.splice(selectedIndex, 1);
                                        }
    
                                        handleAnswerChange(questionIndex, selectedOptions, type);
                                    }}
                                >
                                    {choice}
                                </div>
                            ))}
                        </div>
                    </div>
                );
            case "text":
                return (
                    <div className="question" key={questionIndex}>
                        <p>{questionText}</p>
                        <input
                            type="text"
                            value={answers[categoryIndex][questionIndex]}
                            onChange={(e) => handleAnswerChange(questionIndex, e.target.value, type)}
                        />
                    </div>
                );
            default:
                return null;
        }
    };  

    const calculateProgress = () => {
        let totalQuestionsAnswered = answers.reduce((total, categoryAnswers) => {
            return total + categoryAnswers.filter(answer => answer !== '').length;
        }, 0);

        return Math.round((totalQuestionsAnswered / questions.length) * 100);
    };

    return (
        <div className="questionnaire">
            <div className="progress-bar-container">
                <ProgressBar completed={calculateProgress()} />
            </div>
            <h2>{categories[categoryIndex].toUpperCase()}</h2> 
            {renderQuestions()}
            <div className="pagination">
                <button onClick={prevPage} disabled={currentPage === 1 && categoryIndex === 0}>Previous</button>
                <span>{currentPage} of {totalPages}</span>
                <button onClick={nextPage}>
                    {currentPage === totalPages && categoryIndex === categories.length - 1 ? "Finish" : "Next"}
                </button>
            </div>
        </div>
    );
}

export default Questionnaire;
