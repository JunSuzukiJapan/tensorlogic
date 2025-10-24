// Test entity display with large number of entities

entity Person
entity Country

relation lives_in(person: Person, country: Country)

main {
    // Add many people
    lives_in(alice, japan)
    lives_in(bob, usa)
    lives_in(charlie, uk)
    lives_in(david, france)
    lives_in(eve, germany)
    lives_in(frank, canada)
    lives_in(grace, australia)
    lives_in(henry, brazil)
    lives_in(iris, india)
    lives_in(jack, china)
    lives_in(kate, korea)
    lives_in(leo, mexico)
    lives_in(mary, spain)
    lives_in(nancy, italy)
    lives_in(oscar, russia)

    // Display entities - should show first 10 + count for Person
    with Person {
        // Expected: 15 people, show first 10 + "... and 5 more"
    }

    // Display countries - should show all since <= 10
    with Country {
        // Expected: 15 countries, show first 10 + "... and 5 more"
    }
}
