import React from 'react';
import Questionnaire from '../Questionnaire';
import Footer from '../Footer';

const questions = [
    {
        category: "personality",
        type: "single-select",
        question: "What energy level would your ideal dog have?",
        choices: ["Low Energy (couch potato)", "Moderate Energy (daily walks)", "High Energy (regular, vigorous exercise)", "Extremely High Energy (intense physical/mental stimulation)"],
    },
    {
        category: "personality",
        type: "single-select",
        question: "What would your ideal dog's temperament look like?",
        choices: ["Snugglebugs (gentle, loving)", "Social Butterflies (outgoing, friendly)", "Curious Explorers (curious, inquisitive)", "Zen Masters (calm, composed)", "Watchful Guardians (alert, vigilant)"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "Do you have young children?",
        choices: ["Yes", "No", "Parents-to-be"],
    },
    {
        category: "lifestyle",
        type: "multiple-select",
        question: "What other pets do you own? (Select all that apply)",
        choices: ["Small Dog(s) (< 20 pounds)", "Medium Dog(s) (20 - 50 pounds)", "Large Dog(s) (50 - 100+ pounds)", "Cat", "Small Mammals (rabbits, hamsters, etc.)", "Birds", "Fish", "Reptiles", "Farm Animals"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "How do you feel about grooming and shedding?",
        choices: ["Enjoy grooming, don't mind shedding", "Enjoy grooming, minimal shedding", "Low-maintenance, don't mind shedding", "Low-maintenance, minimal shedding"],
    },
    {
        category: "lifestyle",
        type: "multiple-select",
        question: "What is your primary purpose for getting a dog? (Select all that apply)",
        choices: ["Companionship", "Dog Shows/Competitions", "Dog Sports/Activities"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "What is the climate like where you live?",
        choices: ["Moderate year-round", "Cold climate", "Warm climate", "Willing to accommodate my dog accordingly"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "Are you looking for a hypoallergenic dog?",
        choices: ["Yes", "No"],
    },
    {
        category: "appearance",   
        type: "single-select",
        question: "What size of dog are you drawn to?",
        choices: ["Small (< 20 pounds)", "Medium (20 - 50 pounds)", "Large (50 - 100+ pounds)", "No size preference"],
    },
    {
        category: "appearance", 
        type: "single-select",
        question: "What build of dog are you drawn to?",
        choices: ["Broad & Burly", "Long & Lean", "Build is not a major factor"],
    },
    {
        category: "appearance", 
        type: "single-select",
        question: "Do you prefer long or short coated breeds?",
        choices: ["Long coated", "Short coated", "Coat is not a major factor"],
    },
    {
        category: "appearance", 
        type: "text",
        question: "Are you looking for any distinctive features?",
    },
    {
        category: "personality",
        type: "multiple-select",
        question: "Are you interested in particular drives in a dog? (Select all that apply)",
        choices: ["Prey drive", "Toy drive", "Food drive", "Low drive"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "What is your budget (health conditions and expenses) for dog ownership?",
        choices: ["Willing to invest", "Financially flexible", "Budget-conscious", "Strict budget"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "How much time can you dedicate to training your dog each day?",
        choices: ["Willing to invest time/effort", "Short training sessions", "Prefer an easy-to-train breed"],
    },
    {
        category: "lifestyle",
        type: "single-select",
        question: "What is your living situation like?",
        choices: ["Apartment, limited space", "House, fenced yard", "Rural, plenty of outdoor space", "Flexible to dog's needs"],
    },
    {
        category: "personality",
        type: "single-select",
        question: "Are you concerned about the vocality of your dog?",
        choices: ["Need quiet, non-vocal", "No excessive barking", "Enjoy vocal communication", "No preference/concern"],
    }
]

function Consultation() {
    return (
        <div>
            <Questionnaire questions={questions} />
            <Footer/>
        </div>
    );
}

export default Consultation;